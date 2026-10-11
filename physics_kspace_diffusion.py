import torch
from torch import nn
import torch.nn.functional as F
import fastmri


# ============================================================
# Utility Functions
# ============================================================

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """
    Extract values from a 1-D tensor a at timestep t.

    Parameters
    ----------
    a : Tensor
        Tensor containing timestep-dependent values.
    t : Tensor
        Timestep indices [B].
    x_shape : tuple
        Target tensor shape.

    Returns
    -------
    Tensor
        Reshaped extracted values.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(
        b,
        *((1,) * (len(x_shape) - 1))
    )


def noise_like(shape, device, repeat=False):
    """
    Generate Gaussian noise.
    """

    repeat_noise = lambda: torch.randn(
        (1, *shape[1:]),
        device=device
    ).repeat(
        shape[0],
        *((1,) * (len(shape) - 1))
    )

    noise = lambda: torch.randn(
        shape,
        device=device
    )

    return repeat_noise() if repeat else noise()


# ============================================================
# Cosine Beta Schedule
# ============================================================

def cosine_beta_schedule(
    timesteps,
    s=0.008
):
    """
    Cosine schedule retained for compatibility with
    diffusion implementations.

    Returns
    -------
    Tensor
        Beta schedule.
    """

    import math

    steps = timesteps + 1

    x = torch.linspace(
        0,
        timesteps,
        steps,
        dtype=torch.float64
    )

    alphas_cumprod = torch.cos(
        (
            (x / timesteps) + s
        ) /
        (1 + s) *
        math.pi *
        0.5
    ) ** 2

    alphas_cumprod = (
        alphas_cumprod /
        alphas_cumprod[0]
    )

    betas = 1 - (
        alphas_cumprod[1:] /
        alphas_cumprod[:-1]
    )

    return torch.clip(
        betas,
        0,
        0.999
    )


# ============================================================
# K-Space Cold Diffusion Mask Sequence
# ============================================================

def mask_sequence_sample(mask_fold):
    """
    Generate the masking sequence used by the original
    K-Space Cold Diffusion implementation.

    IMPORTANT:
    - Select ZERO locations from mask_fold.
    - Randomly permute the selected coordinates.

    This must remain identical to the original KCD trajectory.
    Changing == 0 to > 0 changes the degradation process and can
    cause ground-truth leakage during evaluation.
    """

    mask_seqs = []

    for i in range(mask_fold.shape[0]):
        mask_idx = torch.where(mask_fold[i][0] == 0)

        mask_seq = torch.cat(
            (
                mask_idx[0][:, None],
                mask_idx[1][:, None]
            ),
            dim=1
        )

        r = torch.randperm(
            mask_seq.shape[0],
            device=mask_seq.device
        )

        mask_seq = mask_seq[r[:]]
        mask_seqs.append(mask_seq)

    return mask_seqs


# ============================================================
# Physics-Incorporated K-Space Cold Diffusion
# ============================================================

class PhysicsKspaceDiffusion(nn.Module):

    def __init__(
        self,
        denoise_fn,
        image_size,
        device_of_kernel='cuda',
        channels=2,
        timesteps=1000,
        loss_type='l1',
        blur_routine='Constant',
        train_routine='Final',
        sampling_routine='x0_step_down',
        discrete=False,
        dc_weight=1.0,
        debug=False,
        adaptive_residual=False,
        delta_residual_high=0.02,
        delta_residual_low=0.005,
        adaptive_jump_high=1,
        adaptive_jump_medium=5,
        adaptive_jump_low=10
    ):
        """
        Physics-Incorporated K-Space Cold Diffusion.

        The original KCD degradation and reconstruction process
        is retained. During inference, an explicit MRI
        k-space data-consistency projection is inserted.

        Parameters
        ----------
        denoise_fn : nn.Module
            U-Net denoiser.

        image_size : int
            Spatial reconstruction size.

        device_of_kernel : str
            Device used for reconstruction operations.

        channels : int
            Number of image channels. For complex images:
            channels = 2.

        timesteps : int
            Number of KCD diffusion timesteps.

        loss_type : str
            'l1' or 'l2'.

        blur_routine : str
            Original KCD blur routine.

        train_routine : str
            Original KCD training routine.

        sampling_routine : str
            Original KCD sampling routine.

        discrete : bool
            Retained for compatibility.

        dc_weight : float
            Data consistency weight.

            1.0 = hard data consistency.

            This is retained as a parameter so that soft DC
            can be investigated later.

        debug : bool
            Print diagnostic information during sampling.

        adaptive_residual : bool
            Enable residual-based adaptive timestep control during
            inference. When False, the original one-step-at-a-time
            KCD trajectory is retained.

        delta_residual_high : float
            Upper threshold for relative residual change ΔR_t.

        delta_residual_low : float
            Lower threshold for relative residual change ΔR_t.

        adaptive_jump_high : int
            Timestep transition when ΔR_t > delta_residual_high.

        adaptive_jump_medium : int
            Timestep transition when delta_residual_low < ΔR_t <=
            delta_residual_high.

        adaptive_jump_low : int
            Timestep transition when ΔR_t <= delta_residual_low.
        """

        super().__init__()

        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.device_of_kernel = device_of_kernel
        self.num_timesteps = timesteps
        self.loss_type = loss_type
        self.blur_routine = blur_routine
        self.train_routine = train_routine
        self.sampling_routine = sampling_routine
        self.discrete = discrete

        self.dc_weight = dc_weight
        self.debug = debug

        # --------------------------------------------------------
        # Residual-Based Adaptive Diffusion Controller
        # --------------------------------------------------------
        self.adaptive_residual = adaptive_residual
        self.delta_residual_high = delta_residual_high
        self.delta_residual_low = delta_residual_low
        self.adaptive_jump_high = adaptive_jump_high
        self.adaptive_jump_medium = adaptive_jump_medium
        self.adaptive_jump_low = adaptive_jump_low

    # ========================================================
    # KCD Forward Degradation
    # ========================================================

    def q_sample(
        self,
        kspace,
        mask_seqs,
        mask,
        mask_fold,
        t
    ):
        """
        Original K-Space Cold Diffusion degradation.

        Parameters
        ----------
        kspace : Tensor
            [B, Nc, H, W, 2]

        mask_seqs : list
            KCD masking sequence.

        mask : Tensor
            Acquisition mask [B, Nc, H, W].

        mask_fold : Tensor
            Folded KCD mask.

        t : Tensor
            Diffusion timestep [B].

        Returns
        -------
        image_part_masked : Tensor
            [B, Nc, H, W, 2]

        kspace_part_masked : Tensor
            [B, Nc, H, W, 2]
        """

        patch_size_H = (
            mask.shape[-2] //
            mask_fold.shape[-2]
        )

        patch_size_W = (
            mask.shape[-1] //
            mask_fold.shape[-1]
        )

        mask_part = torch.ones(
            mask_fold.shape,
            device=mask_fold.device,
            dtype=mask_fold.dtype
        )

        for i in range(mask_fold.shape[0]):

            mask_seq = mask_seqs[i]

            step = (
                mask_seq.shape[0] /
                self.num_timesteps
            )

            num_points = int(
                t[i].item() * step
            )

            if num_points > 0:

                coords = mask_seq[
                    :num_points
                ]

                mask_part[
                    i,
                    ...,
                    coords[:, 0],
                    coords[:, 1]
                ] = 0.0

        mask_expand = F.interpolate(
            mask_part,
            scale_factor=[
                patch_size_H,
                patch_size_W
            ],
            mode='nearest'
        ).to(self.device_of_kernel)

        kspace_part_masked = (
            kspace *
            mask_expand.unsqueeze(-1).repeat(
                1,
                kspace.shape[1],
                1,
                1,
                kspace.shape[-1]
            )
        )

        image_part_masked = fastmri.ifft2c(
            kspace_part_masked
        )

        return (
            image_part_masked,
            kspace_part_masked
        )

    # ========================================================
    # Physics Data Consistency
    # ========================================================

    def data_consistency(
        self,
        predicted_kspace,
        measured_kspace,
        mask
    ):
        """
        Hard MRI k-space data consistency.

        Implements:

            k_DC =
                M * y
                +
                (1-M) * k_pred

        where:

            y     = measured undersampled k-space
            M     = acquisition mask
            k_pred = predicted k-space

        Parameters
        ----------
        predicted_kspace : Tensor
            [B, Nc, H, W, 2]

        measured_kspace : Tensor
            [B, Nc, H, W, 2]

        mask : Tensor
            [B, Nc, H, W]

        Returns
        -------
        Tensor
            Data-consistent k-space.
        """

        assert predicted_kspace.shape == measured_kspace.shape, (
            "predicted_kspace and measured_kspace must have "
            "the same shape. "
            f"Got {predicted_kspace.shape} and "
            f"{measured_kspace.shape}"
        )

        B, Nc, H, W, C = predicted_kspace.shape

        assert C == 2, (
            f"Expected complex dimension 2, got {C}"
        )

        assert mask.shape == (
            B,
            Nc,
            H,
            W
        ), (
            f"Expected mask shape "
            f"{(B, Nc, H, W)}, got {mask.shape}"
        )

        # [B, Nc, H, W]
        #      ↓ unsqueeze
        #
        # [B, Nc, H, W, 1]
        mask_complex = mask.unsqueeze(-1).to(
            predicted_kspace.dtype
        )

        # ----------------------------------------------------
        # Hard data consistency
        # ----------------------------------------------------

        k_dc = (
            mask_complex * measured_kspace
            +
            (1.0 - mask_complex) *
            predicted_kspace
        )

        return k_dc

    # ========================================================
    # Diagnostic Check
    # ========================================================

    def check_measurement_consistency(
        self,
        kspace,
        measured_kspace,
        mask
    ):
        """
        Verify:

            measured_kspace = mask * kspace

        This should be approximately zero error for the
        current PhysicsDiffusion transform.
        """

        mask_complex = mask.unsqueeze(-1).to(
            kspace.dtype
        )

        expected_measured = (
            mask_complex *
            kspace
        )

        error = torch.max(
            torch.abs(
                measured_kspace -
                expected_measured
            )
        )

        if self.debug:

            print(
                "Maximum measured-kspace error:",
                error.item()
            )

            print(
                "kspace_full abs max:",
                torch.abs(kspace).max().item()
            )

            print(
                "measured_kspace abs max:",
                torch.abs(
                    measured_kspace
                ).max().item()
            )

            print(
                "mask mean:",
                mask.float().mean().item()
            )

        return error

    # ========================================================
    # Residual-Based Adaptive Diffusion Controller
    # ========================================================

    def reconstruction_residual(self, new_img, old_img, eps=1e-8):
        """Compute normalized reconstruction residual R_t."""
        diff = new_img - old_img

        diff_norm = torch.sqrt(
            torch.sum(diff ** 2, dim=tuple(range(1, diff.ndim))) + eps
        )
        old_norm = torch.sqrt(
            torch.sum(old_img ** 2, dim=tuple(range(1, old_img.ndim))) + eps
        )

        return diff_norm / old_norm

    def relative_residual_change(
        self,
        current_residual,
        previous_residual,
        eps=1e-8
    ):
        """
        Compute relative residual change ΔR_t.

        ΔR_t = |R_t - R_prev| / (R_prev + eps)
        """
        return torch.abs(current_residual - previous_residual) / (
            previous_residual + eps
        )

    def adaptive_timestep_jump(self, delta_residual):
        """
        Discrete residual-change controller.

        ΔR_t > 0.02                  -> timestep transition = 1
        0.005 < ΔR_t <= 0.02        -> timestep transition = 5
        ΔR_t <= 0.005                -> timestep transition = 10
        """
        delta = float(delta_residual)

        if delta > self.delta_residual_high:
            return self.adaptive_jump_high
        elif delta > self.delta_residual_low:
            return self.adaptive_jump_medium
        else:
            return self.adaptive_jump_low

    # ========================================================
    # Physics KCD Sampling
    # ========================================================

    @torch.no_grad()
    def sample(
        self,
        kspace,
        mask,
        mask_fold,
        t,
        measured_kspace=None
    ):
        """
        Physics-incorporated reverse KCD sampling with optional
        residual-based adaptive timestep control.

        Adaptive mechanism
        ------------------
        At each reverse iteration:

            1. Denoise the current image at timestep t.
            2. Transform the prediction to k-space.
            3. Apply hard MRI data consistency.
            4. Compute the normalized residual R_t.
            5. Compute the relative residual change ΔR_t.
            6. Select the timestep transition using:

                   ΔR_t > 0.02           -> Δt = 1
                   0.005 < ΔR_t <= 0.02 -> Δt = 5
                   ΔR_t <= 0.005         -> Δt = 10

            7. Move from t -> t_next using the generalized KCD
               x0-step-down rule.

        The first adaptive iteration has no previous residual, so it
        conservatively uses Δt = 1.

        Setting adaptive_residual=False reproduces the original
        one-timestep-at-a-time sampling trajectory.

        Returns
        -------
        xt : Tensor
            Initial degraded image.

        direct_recons : Tensor
            First direct denoised reconstruction.

        img : Tensor
            Final reconstruction.
        """

        self.denoise_fn.eval()

        # ----------------------------------------------------
        # Shape validation
        # ----------------------------------------------------

        batch_size, Nc, H, W, C = kspace.shape

        assert H == self.image_size, (
            f"Expected H={self.image_size}, got {H}"
        )

        assert W == self.image_size, (
            f"Expected W={self.image_size}, got {W}"
        )

        assert C == 2, (
            f"Expected complex dimension 2, got {C}"
        )

        if measured_kspace is not None:

            assert measured_kspace.shape == kspace.shape, (
                "measured_kspace shape does not match "
                f"kspace: {measured_kspace.shape} "
                f"vs {kspace.shape}"
            )

            assert mask.shape == (
                batch_size,
                Nc,
                H,
                W
            ), (
                f"Unexpected mask shape: "
                f"{mask.shape}"
            )

            # ------------------------------------------------
            # Verify measurement consistency once.
            # ------------------------------------------------

            self.check_measurement_consistency(
                kspace,
                measured_kspace,
                mask
            )

        # ----------------------------------------------------
        # Initial timestep
        # ----------------------------------------------------

        if t is None:
            t = self.num_timesteps

        t = int(t)
        t = max(1, min(t, self.num_timesteps))

        # ----------------------------------------------------
        # Generate KCD masking sequence
        # ----------------------------------------------------

        mask_seqs = mask_sequence_sample(
            mask_fold
        )

        # ----------------------------------------------------
        # Initial KCD degradation
        # ----------------------------------------------------

        t_tensor = torch.full(
            (batch_size,),
            t,
            dtype=torch.long,
            device=kspace.device
        )

        img, kspace_current = self.q_sample(
            kspace,
            mask_seqs,
            mask,
            mask_fold,
            t_tensor
        )

        xt = img
        direct_recons = None

        # Previous residual used to calculate relative residual change.
        previous_residual = None

        # Keep a record when debug mode is enabled.
        adaptive_history = []

        # ----------------------------------------------------
        # Reverse KCD sampling
        # ----------------------------------------------------

        while t > 0:

            step = torch.full(
                (batch_size,),
                t,
                dtype=torch.long,
                device=kspace.device
            )

            img_before_update = img

            # ------------------------------------------------
            # Denoising
            # ------------------------------------------------

            x = torch.zeros_like(
                img
            ).to(
                self.device_of_kernel
            )

            for i in range(Nc):

                x[:, i, :] = (
                    self.denoise_fn(
                        img[:, i, :]
                        .permute(0, 3, 1, 2),
                        step
                    )
                    .permute(0, 2, 3, 1)
                )

            # ------------------------------------------------
            # Image -> k-space
            # ------------------------------------------------

            k_pred = fastmri.fft2c(x)

            # ------------------------------------------------
            # Physics incorporation
            # ------------------------------------------------

            if measured_kspace is not None:

                k_current = self.data_consistency(
                    predicted_kspace=k_pred,
                    measured_kspace=measured_kspace,
                    mask=mask
                )

                # Convert corrected k-space back to image domain.
                x = fastmri.ifft2c(
                    k_current
                )

            else:

                # Original KCD path.
                k_current = k_pred

            # ------------------------------------------------
            # Direct reconstruction
            # ------------------------------------------------

            if direct_recons is None:
                direct_recons = x

            # ------------------------------------------------
            # Residual measurement and adaptive controller
            # ------------------------------------------------
            # R_t measures the normalized change in the reconstruction.
            # ΔR_t measures the relative change in R_t between two
            # consecutive reverse-diffusion iterations.
            # ------------------------------------------------

            residual = self.reconstruction_residual(
                x,
                img_before_update
            )

            residual_value = residual.mean().item()

            if previous_residual is None:
                # No previous residual exists on the first iteration.
                # Use the conservative transition specified by the
                # controller: Δt = 1.
                delta_residual_value = None
            else:
                delta_residual = self.relative_residual_change(
                    residual,
                    previous_residual
                )
                delta_residual_value = delta_residual.mean().item()

            if self.adaptive_residual and delta_residual_value is not None:
                jump = self.adaptive_timestep_jump(
                    delta_residual_value
                )
            else:
                jump = 1

            # Never skip past timestep zero.
            next_t = max(0, t - jump)

            adaptive_history.append(
                {
                    "t": t,
                    "next_t": next_t,
                    "jump": jump,
                    "residual": residual_value,
                    "delta_residual": delta_residual_value
                }
            )

            if self.debug:
                delta_text = (
                    "N/A" if delta_residual_value is None
                    else f"{delta_residual_value:.6f}"
                )

                print(
                    f"[Adaptive KCD] "
                    f"t={t:4d} -> {next_t:4d} | "
                    f"Δt={jump:2d} | "
                    f"R_t={residual_value:.6f} | "
                    f"ΔR_t={delta_text}"
                )

            # Save the current residual for the next reverse step.
            previous_residual = residual.detach()

            # ------------------------------------------------
            # KCD x0-step-down
            # ------------------------------------------------
            #
            # Generalized from:
            #
            #     x = img - x_t + x_{t-1}
            #
            # to:
            #
            #     x = img - x_t + x_{t_next}
            #
            # This lets the residual controller skip multiple
            # diffusion states when the reconstruction is stable.
            # ------------------------------------------------

            if self.sampling_routine == 'x0_step_down':

                k_times = k_current

                with torch.no_grad():

                    x_times, _ = self.q_sample(
                        k_times,
                        mask_seqs,
                        mask,
                        mask_fold,
                        step
                    )

                next_step = torch.full(
                    (batch_size,),
                    next_t,
                    dtype=torch.long,
                    device=kspace.device
                )

                with torch.no_grad():

                    x_times_sub_1, _ = self.q_sample(
                        k_times,
                        mask_seqs,
                        mask,
                        mask_fold,
                        next_step
                    )

                x = (
                    img_before_update -
                    x_times +
                    x_times_sub_1
                )

                # ------------------------------------------------
                # Re-apply physics after the adaptive KCD step.
                # ------------------------------------------------

                if measured_kspace is not None:

                    k_adaptive = fastmri.fft2c(x)

                    k_adaptive = self.data_consistency(
                        predicted_kspace=k_adaptive,
                        measured_kspace=measured_kspace,
                        mask=mask
                    )

                    x = fastmri.ifft2c(
                        k_adaptive
                    )

                    k_current = k_adaptive

            elif self.sampling_routine == 'default':

                # The original default KCD branch is retained.
                # For adaptive mode, use the selected next timestep.
                target_t = next_t

                if self.blur_routine == 'Individual_Incremental':

                    target_t = max(0, target_t - 1)

                target_step = torch.full(
                    (batch_size,),
                    target_t,
                    dtype=torch.long,
                    device=kspace.device
                )

                with torch.no_grad():

                    x, _ = self.q_sample(
                        k_current,
                        mask_seqs,
                        mask,
                        mask_fold,
                        target_step
                    )

            else:

                raise NotImplementedError(
                    f"Unknown sampling routine: "
                    f"{self.sampling_routine}"
                )

            # ------------------------------------------------
            # Final state update
            # ------------------------------------------------

            img = x
            t = next_t

        self.denoise_fn.train()

        # Make the adaptive trajectory available to test.py for
        # NFE/step-size analysis without changing the public return
        # signature of sample().
        self.last_adaptive_history = adaptive_history

        if self.debug and self.adaptive_residual:

            print(
                "[Adaptive KCD] "
                f"Completed in {len(adaptive_history)} "
                f"reverse iterations."
            )

        return (
            xt,
            direct_recons,
            img
        )

    # ========================================================
    # Training Loss
    # ========================================================

    def p_losses(
        self,
        kspace,
        mask_seqs,
        mask,
        mask_fold,
        t
    ):
        """
        Original KCD training objective.

        Physics data consistency is NOT included in the
        training loss in this first ablation.

        This is intentional so that:

            KCD
            vs.
            Physics-KCD

        differs only in the inference-time physics operation.
        """

        B, Nc, H, W, C = kspace.shape

        # ----------------------------------------------------
        # Fully sampled target image
        # ----------------------------------------------------

        x_start = fastmri.ifft2c(
            kspace
        )

        # ----------------------------------------------------
        # KCD degradation
        # ----------------------------------------------------

        if self.train_routine == 'Final':

            x_blur, _ = self.q_sample(
                kspace,
                mask_seqs,
                mask,
                mask_fold,
                t
            )

            # ------------------------------------------------
            # Denoising
            # ------------------------------------------------

            x_recon = torch.zeros_like(
                x_blur
            ).to(
                self.device_of_kernel
            )

            for i in range(Nc):

                x_recon[:, i, :] = (
                    self.denoise_fn(
                        x_blur[:, i, :]
                        .permute(0, 3, 1, 2),
                        t
                    )
                    .permute(0, 2, 3, 1)
                )

            # ------------------------------------------------
            # Original KCD loss
            # ------------------------------------------------

            if self.loss_type == 'l1':

                loss = (
                    x_start -
                    x_recon
                ).abs().mean()

            elif self.loss_type == 'l2':

                loss = F.mse_loss(
                    x_start,
                    x_recon
                )

            else:

                raise NotImplementedError(
                    f"Unknown loss type: "
                    f"{self.loss_type}"
                )

        else:

            raise NotImplementedError(
                "Only train_routine='Final' "
                "is implemented."
            )

        return loss

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        kspace,
        mask,
        mask_fold,
        *args,
        **kwargs
    ):
        """
        Training forward pass.

        Input:

            kspace [B, Nc, H, W, 2]

        Note:
            measured_kspace is not required during training
            for the first Physics-KCD experiment.
        """

        B, Nc, H, W, C = kspace.shape

        assert H == self.image_size
        assert W == self.image_size
        assert C == 2

        device = kspace.device

        # ----------------------------------------------------
        # Random KCD timestep
        # ----------------------------------------------------

        t = torch.randint(
            0,
            self.num_timesteps + 1,
            (B,),
            device=device
        ).long()

        # ----------------------------------------------------
        # KCD masking sequence
        # ----------------------------------------------------

        mask_seqs = mask_sequence_sample(
            mask_fold
        )

        # ----------------------------------------------------
        # Original KCD training loss
        # ----------------------------------------------------

        return self.p_losses(
            kspace,
            mask_seqs,
            mask,
            mask_fold,
            t
        )