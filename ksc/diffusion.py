"""K-space cold diffusion: degradation, training objective and the three samplers.

Notation (paper Sec. 3):
  k        fully sampled k-space,  M  acquisition mask,  y = M * k
  D_t(k)   KCD degradation: removes the first floor(t * n_u / T) unsampled columns
           (random but fixed order per slice); D_0 = identity, D_T(k) = y.
  f_theta  U-Net predicting the clean image x_0 from the degraded image x_t.

Models (all sample with the x0-step-down rule  x_next = x_t - D_t(k0) + D_next(k0)):
  KSC-Diff        trained without DC; fixed dt = 1, no DC
  PhyKSC-Diff     trained with a hard DC layer  k0 <- M*y + (1-M)*k0  on the prediction;
                  the same DC layer is kept in its forward pass at inference; fixed dt = 1
  RA-PhyKSC-Diff  PhyKSC-Diff network + DC + residual-guided jump  dt in {1, 5, 10} grid levels

Note: D_t never removes acquired columns, so in x0-step-down sampling the acquired columns of
x_t stay equal to y and DC applied at inference does not change the trajectory. DC therefore
affects KSC vs PhyKSC only through training, and RA through its residual signal.
"""
import torch
import torch.nn.functional as F

from .data import fft2c, ifft2c


def to_ch(x):   # complex [B,H,W] -> real [B,2,H,W]
    return torch.stack((x.real, x.imag), dim=1)


def to_cplx(x):  # real [B,2,H,W] -> complex [B,H,W]
    return torch.complex(x[:, 0], x[:, 1])


class KSpaceColdDiffusion:
    def __init__(self, model, num_timesteps=1000):
        self.model = model
        self.T = num_timesteps

    # ---------------------------------------------------------------- degradation
    def degrade_mask(self, order, mask, t):
        """Column mask of D_t.  order: [B,W] removal rank (inf for sampled); t: [B]."""
        n_unsampled = (mask == 0).sum(dim=1).float()                  # [B]
        n_removed = torch.floor(t.float() * n_unsampled / self.T)     # [B]
        keep = (order >= n_removed[:, None]).float()                  # [B,W]
        return keep[:, None, :]                                       # [B,1,W] broadcast over rows

    def degrade(self, k, order, mask, t):
        return k * self.degrade_mask(order, mask, t)

    # ---------------------------------------------------------------- training
    def loss(self, k, mask, order, dc_train=False):
        """L1 x0-prediction loss.  dc_train=True (PhyKSC-Diff) passes the prediction through the
        hard DC layer  k_hat <- M*y + (1-M)*k_hat  before the loss, so the network is trained on
        the data-consistent reconstruction and only the unacquired columns carry gradient."""
        b = k.shape[0]
        t = torch.randint(0, self.T + 1, (b,), device=k.device)
        x_t = ifft2c(self.degrade(k, order, mask, t))
        x0_hat = self.model(to_ch(x_t), t)
        if dc_train:
            k_hat = self.dc(fft2c(to_cplx(x0_hat.float())), k * mask[:, None, :], mask)
            x0_hat = to_ch(ifft2c(k_hat))
        return F.l1_loss(x0_hat.float(), to_ch(ifft2c(k)))

    # ---------------------------------------------------------------- inference
    @staticmethod
    def dc(k_pred, y, mask):
        m = mask[:, None, :]
        return m * y + (1 - m) * k_pred

    def predict(self, x_t, t):
        with torch.autocast("cuda", dtype=torch.bfloat16):  # same precision as training
            out = self.model(to_ch(x_t), t)
        return to_cplx(out.float())

    @torch.no_grad()
    def sample(self, k_full, mask, order, steps=1000, physics=True, adaptive=False,
               thr_high=0.02, thr_low=0.005, jumps=(1, 5, 10), eps=1e-8, record=False,
               schedule=None, probe_jumps=()):
        """Reverse KCD from t=T (pure undersampled measurement) to t=0.

        `steps` sets the diffusion-time configuration: the reverse trajectory is the grid
        t_i = round(T*(steps-i)/steps), i=0..steps, so the fixed sampler uses NFE = steps.
        Adaptive jumps are expressed in grid levels (for steps=T they are raw timesteps).
        Each slice in the batch keeps its own position, residual and jump history.

        schedule     explicit list of grid jumps (overrides the controller; used for the
                     random-order control). Must sum to `steps`.
        probe_jumps  (batch size 1, fixed dt=1 path) at every state, also form the state a jump
                     of j grid levels would reach, and when the trajectory arrives there report
                     the relative deviation ||x_jump - x_fixed|| / ||x_fixed||  (RQ2 probe).

        Returns (final image [B,H,W] complex, nfe [B], history per slice, probe rows).
        history rows: (t, t_next, R_t, dR_t); probe rows: (t, R_t, dR_t, {j: deviation}).
        """
        B = k_full.shape[0]
        dev = k_full.device
        y = k_full * mask[:, None, :]
        grid = torch.round(torch.linspace(self.T, 0, steps + 1, device=dev)).long()
        assert schedule is None or sum(schedule) == steps
        assert not probe_jumps or (B == 1 and not adaptive and schedule is None)

        pos = torch.zeros(B, dtype=torch.long, device=dev)        # index into grid
        img = ifft2c(self.degrade(k_full, order, mask, grid[pos]))  # == zero-filled y
        prev_R = torch.full((B,), float("nan"), device=dev)
        nfe = torch.zeros(B, dtype=torch.long, device=dev)
        history = [[] for _ in range(B)] if record else None
        pending, probe = {}, {}   # probe: grid pos -> partial row; pending: target pos -> [(src pos, j, x)]
        it = 0

        while True:
            active = (pos < steps).nonzero(as_tuple=True)[0]
            if active.numel() == 0:
                break
            a_img, a_mask, a_order = img[active], mask[active], order[active]
            t = grid[pos[active]]

            if probe_jumps:
                p = int(pos[0])
                for src, j, xj in pending.pop(p, []):
                    probe[src][3][j] = float((xj - a_img).norm() / a_img.norm())

            x0 = self.predict(a_img, t)
            nfe[active] += 1
            k0 = fft2c(x0)
            if physics:
                k0 = self.dc(k0, y[active], a_mask)

            # residual-change signal (paper Sec. 3.3), computed whenever it is used or recorded
            if adaptive or record or probe_jumps:
                x_dc = ifft2c(k0)
                R = (x_dc - a_img).flatten(1).norm(dim=1) / (a_img.flatten(1).norm(dim=1) + eps)
                pR = prev_R[active]
                dR = (R - pR).abs() / (pR + eps)
                prev_R[active] = R
            else:
                R = dR = None

            if schedule is not None:
                jump = torch.full_like(t, schedule[it])
            elif adaptive:
                jump = torch.where(dR > thr_high, jumps[0],
                                   torch.where(dR > thr_low, jumps[1], jumps[2]))
                jump = torch.where(torch.isnan(pR), torch.ones_like(jump), jump)  # first step: dt=1
            else:
                jump = torch.ones_like(t)

            new_pos = torch.clamp(pos[active] + jump, max=steps)
            t_next = grid[new_pos]
            base = a_img - ifft2c(self.degrade(k0, a_order, a_mask, t))
            x_next = base + ifft2c(self.degrade(k0, a_order, a_mask, t_next))
            if physics:  # second hard DC projection before the next reverse iteration
                x_next = ifft2c(self.dc(fft2c(x_next), y[active], a_mask))

            if probe_jumps:
                p = int(pos[0])
                probe[p] = [int(t[0]), float(R[0]), None if torch.isnan(dR[0]) else float(dR[0]), {}]
                for j in probe_jumps:
                    if p + j <= steps:
                        xj = base + ifft2c(self.degrade(k0, a_order, a_mask, grid[p + j][None]))
                        if physics:
                            xj = ifft2c(self.dc(fft2c(xj), y, a_mask))
                        pending.setdefault(p + j, []).append((p, j, xj))

            img[active] = x_next
            if record:
                for j, b in enumerate(active.tolist()):
                    history[b].append((int(t[j]), int(t_next[j]),
                                       None if R is None else float(R[j]),
                                       None if dR is None or torch.isnan(dR[j]) else float(dR[j])))
            pos[active] = new_pos
            it += 1

        if probe_jumps:  # jumps landing on the final state t=0
            for src, j, xj in pending.pop(steps, []):
                probe[src][3][j] = float((xj - img[0]).norm() / img[0].norm())
        return img, nfe, history, [probe[p] for p in sorted(probe)]
