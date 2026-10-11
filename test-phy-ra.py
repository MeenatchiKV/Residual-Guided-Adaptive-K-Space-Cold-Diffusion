# ============================================================
# test.py
# Residual-Adaptive Physics-Guided K-Space Cold Diffusion
# RA-PhyKSC-Diff
# Single-Coil fastMRI Knee
#
# Adaptive controller:
#
#   ΔR_t > 0.02                  -> Δt = 1
#   0.005 < ΔR_t <= 0.02       -> Δt = 5
#   ΔR_t <= 0.005               -> Δt = 10
#
# Physics data consistency is enforced during every reverse step.
#
# Visualization and CSV:
#   - Every slice in every batch is visualized.
#   - One CSV row is created for every slice.
#   - Per-sample CSV is saved after every batch.
#   - Adaptive trajectory CSV is saved after every batch.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import pathlib
import time
import random
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import fastmri

from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================
# DATASET AND TRANSFORMS
# ============================================================

from utils.mri_data import SliceDataset
from utils.data_transform import DataTransform_PhysicsDiffusion
from utils.sample_mask import RandomMaskDiffusion

from utils.evaluation_utils import (
    calc_nmse_tensor,
    calc_psnr_tensor,
    calc_ssim_tensor,
)


# ============================================================
# MODEL
# ============================================================

from net.u_net_diffusion import Unet
from physics_kspace_diffusion import PhysicsKspaceDiffusion


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ------------------------------------------------------------
# Image and MRI settings
# ------------------------------------------------------------

IMAGE_SIZE = 320

ACCELERATION = 4

CENTER_FRACTION = 0.08

TIMESTEPS = 200


# ------------------------------------------------------------
# Model settings
# ------------------------------------------------------------

CH_MID = 64


# ------------------------------------------------------------
# DataLoader settings
# ------------------------------------------------------------

BATCH_SIZE = 6

NUM_WORKERS = 0


# ------------------------------------------------------------
# Test dataset
# ------------------------------------------------------------

TEST_PATH = (
    r"/datasets/tir-dataset-260918100010/singlecoil/test"
)


# ------------------------------------------------------------
# Trained RA-PhyKSC-Diff checkpoint
# ------------------------------------------------------------

CHECKPOINT = (
    "/home/jovyan/mri-recon-phy/"
    "saved_models/fastmri_knee/"
    "physics_diffusion_fastmri_4x_T1000_S12420/"
    "model_12420.pt"
)


# ------------------------------------------------------------
# Results directory
# ------------------------------------------------------------

RESULTS_DIR = (
    f"/home/jovyan/mri-recon-ra/"
    f"saved_models/adaptive_test_results_t{TIMESTEPS}"
)


# ============================================================
# ADAPTIVE STRATEGY
# ============================================================

DELTA_R_HIGH = 0.02

DELTA_R_LOW = 0.005

JUMP_HIGH = 1

JUMP_MEDIUM = 5

JUMP_LOW = 10


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILE PATHS
# ============================================================

RESULTS_CSV = os.path.join(
    RESULTS_DIR,
    "ra_physics_kcd_test_results.csv"
)


ADAPTIVE_CSV = os.path.join(
    RESULTS_DIR,
    "adaptive_trajectory.csv"
)


STEP_DISTRIBUTION_CSV = os.path.join(
    RESULTS_DIR,
    "adaptive_step_distribution.csv"
)


SUMMARY_CSV = os.path.join(
    RESULTS_DIR,
    "ra_physics_kcd_summary.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 90)

print(
    "RA-PhyKSC-Diff TESTING"
)

print("=" * 90)

print(
    "Device        :",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU           :",
        torch.cuda.get_device_name(0)
    )

print(
    "Image size    :",
    IMAGE_SIZE
)

print(
    "Acceleration  : R" +
    str(ACCELERATION)
)

print(
    "Center fraction:",
    CENTER_FRACTION
)

print(
    "Timesteps     :",
    TIMESTEPS
)

print(
    "Batch size    :",
    BATCH_SIZE
)

print(
    "Checkpoint    :",
    CHECKPOINT
)

print(
    "Results       :",
    RESULTS_DIR
)

print("=" * 90)

print(
    "Adaptive controller:"
)

print(
    "  ΔR > 0.020                  -> Δt = 1"
)

print(
    "  0.005 < ΔR <= 0.020       -> Δt = 5"
)

print(
    "  ΔR <= 0.005               -> Δt = 10"
)

print("=" * 90)


# ============================================================
# DATA
# ============================================================

mask_func = RandomMaskDiffusion(
    acceleration=ACCELERATION,
    center_fraction=CENTER_FRACTION,
    size=(
        1,
        IMAGE_SIZE,
        IMAGE_SIZE
    ),
)


data_transform = DataTransform_PhysicsDiffusion(
    mask_func,
    img_size=IMAGE_SIZE,
    flag_singlecoil=True,
)


test_dataset = SliceDataset(
    root=pathlib.Path(TEST_PATH),
    transform=data_transform,
    challenge="singlecoil",
    num_skip_slice=5,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


print(
    "Test slices :",
    len(test_dataset)
)

print(
    "Test batches:",
    len(test_loader)
)


# ============================================================
# MODEL
# ============================================================

model = Unet(
    dim=CH_MID,
    dim_mults=(
        1,
        2,
        4,
        8
    ),
    channels=2,
).to(DEVICE)


# ============================================================
# PHYSICS-KCD DIFFUSION MODEL
# ============================================================

diffusion = PhysicsKspaceDiffusion(
    model,

    image_size=IMAGE_SIZE,

    device_of_kernel=DEVICE,

    channels=2,

    timesteps=TIMESTEPS,

    loss_type="l1",

    blur_routine="Constant",

    train_routine="Final",

    sampling_routine="x0_step_down",

    discrete=False,

    dc_weight=1.0,

    debug=False,

    adaptive_residual=True,

    delta_residual_high=DELTA_R_HIGH,

    delta_residual_low=DELTA_R_LOW,

    adaptive_jump_high=JUMP_HIGH,

    adaptive_jump_medium=JUMP_MEDIUM,

    adaptive_jump_low=JUMP_LOW,

).to(DEVICE)


# ============================================================
# LOAD EMA CHECKPOINT
# ============================================================

if not os.path.isfile(CHECKPOINT):

    raise FileNotFoundError(
        "Checkpoint not found:\n"
        + CHECKPOINT
        + "\n\n"
        "Set CHECKPOINT to the correct "
        "trained checkpoint."
    )


print(
    "\nLoading checkpoint..."
)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)


print(
    "Checkpoint keys:",
    list(checkpoint.keys())
)


# ------------------------------------------------------------
# Determine checkpoint weights
# ------------------------------------------------------------

if "ema" in checkpoint:

    state_dict = checkpoint["ema"]

elif "model" in checkpoint:

    print(
        "WARNING: EMA weights not found; "
        "using model weights."
    )

    state_dict = checkpoint["model"]

elif "model_state_dict" in checkpoint:

    print(
        "WARNING: legacy checkpoint detected; "
        "using model_state_dict."
    )

    state_dict = checkpoint["model_state_dict"]

else:

    state_dict = checkpoint


# ------------------------------------------------------------
# Remap legacy U-Net-only checkpoints if required
# ------------------------------------------------------------

model_state = diffusion.state_dict()

remapped = {}


for key, value in state_dict.items():

    new_key = key

    if (
        new_key not in model_state
        and
        ("denoise_fn." + new_key)
        in model_state
    ):

        new_key = (
            "denoise_fn."
            + new_key
        )

    remapped[new_key] = value


# ------------------------------------------------------------
# Load weights
# ------------------------------------------------------------

missing, unexpected = (
    diffusion.load_state_dict(
        remapped,
        strict=False
    )
)


print(
    "Missing keys    :",
    len(missing)
)

print(
    "Unexpected keys :",
    len(unexpected)
)


if missing:

    print(
        "First missing keys:"
    )

    for key in missing[:10]:

        print(
            "  ",
            key
        )


if unexpected:

    print(
        "First unexpected keys:"
    )

    for key in unexpected[:10]:

        print(
            "First unexpected keys:"
        )

    for key in unexpected[:10]:

        print(
            "  ",
            key
        )


if len(missing) > 0:

    raise RuntimeError(
        "Checkpoint could not be loaded completely. "
        "Check that the U-Net architecture and "
        "diffusion configuration match the "
        "training checkpoint."
    )


diffusion.eval()


print(
    "EMA/model weights loaded successfully."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def scalar_value(x):

    """
    Convert a tensor/scalar metric to Python float.
    """

    if torch.is_tensor(x):

        return float(
            x.detach()
            .cpu()
            .item()
        )

    return float(x)


# ------------------------------------------------------------

def to_magnitude(x):

    """
    Convert complex representation:

        [B, Nc, H, W, 2]

    into magnitude:

        [B, Nc, H, W]
    """

    return fastmri.complex_abs(x)


# ------------------------------------------------------------

def save_image(
    image,
    title,
    filename
):

    """
    Save one reconstruction visualization.
    """

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        image,
        cmap="gray"
    )

    plt.axis("off")

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# TEST LOOP
# ============================================================

results = []

adaptive_rows = []


# ------------------------------------------------------------
# Global counters
# ------------------------------------------------------------

sample_count = 0

total_time = 0.0

total_nfe = 0


print(
    "\nStarting adaptive reconstruction...\n"
)


with torch.no_grad():

    # ========================================================
    # BATCH LOOP
    # ========================================================

    for batch_idx, batch in enumerate(
        tqdm(
            test_loader,
            desc="Testing"
        )
    ):

        # ----------------------------------------------------
        # Support both transform formats
        #
        # 4 outputs:
        #   full kspace
        #   measured kspace
        #   mask
        #   mask_fold
        #
        # 3 outputs:
        #   full kspace
        #   mask
        #   mask_fold
        # ----------------------------------------------------

        if len(batch) == 4:

            (
                kspace_full,
                measured_kspace,
                mask,
                mask_fold
            ) = batch


        elif len(batch) == 3:

            (
                kspace_full,
                mask,
                mask_fold
            ) = batch


            measured_kspace = (
                mask.unsqueeze(-1)
                .to(kspace_full.dtype)
                *
                kspace_full
            )


        else:

            raise RuntimeError(
                f"Unexpected test batch length: "
                f"{len(batch)}."
            )


        # ====================================================
        # MOVE TO DEVICE
        # ====================================================

        kspace_full = (
            kspace_full
            .to(
                DEVICE,
                non_blocking=True
            )
            .float()
        )


        measured_kspace = (
            measured_kspace
            .to(
                DEVICE,
                non_blocking=True
            )
            .float()
        )


        mask = (
            mask
            .to(
                DEVICE,
                non_blocking=True
            )
            .float()
        )


        mask_fold = (
            mask_fold
            .to(
                DEVICE,
                non_blocking=True
            )
            .float()
        )


        # ====================================================
        # BATCH DIMENSIONS
        # ====================================================

        B, Nc, H, W, C = (
            kspace_full.shape
        )


        # ====================================================
        # SHAPE CHECK
        # ====================================================

        assert (
            H,
            W,
            C
        ) == (
            IMAGE_SIZE,
            IMAGE_SIZE,
            2
        ), (
            f"Unexpected k-space shape: "
            f"{kspace_full.shape}"
        )


        # ====================================================
        # MEASUREMENT CONSISTENCY
        # ====================================================

        expected_measured = (
            mask.unsqueeze(-1)
            .to(kspace_full.dtype)
            *
            kspace_full
        )


        measurement_error = (
            torch.max(
                torch.abs(
                    measured_kspace
                    -
                    expected_measured
                )
            )
            .item()
        )


        # ====================================================
        # GROUND TRUTH
        # ====================================================

        gt_complex = fastmri.ifft2c(
            kspace_full
        )


        gt_abs = to_magnitude(
            gt_complex
        )


        # ====================================================
        # ADAPTIVE RECONSTRUCTION
        # ====================================================

        if torch.cuda.is_available():

            torch.cuda.synchronize()


        start = time.perf_counter()


        (
            xt,
            direct_recons,
            reconstruction
        ) = diffusion.sample(

            kspace_full,

            mask,

            mask_fold,

            t=TIMESTEPS,

            measured_kspace=measured_kspace,
        )


        if torch.cuda.is_available():

            torch.cuda.synchronize()


        elapsed = (
            time.perf_counter()
            -
            start
        )


        total_time += elapsed


        # ====================================================
        # RECONSTRUCTION MAGNITUDES
        # ====================================================

        recon_abs = to_magnitude(
            reconstruction
        )


        xt_abs = to_magnitude(
            xt
        )


        direct_abs = to_magnitude(
            direct_recons
        )


        # ====================================================
        # SINGLE COIL
        #
        # Nc = 1
        # ====================================================

        gt_abs_sc = (
            gt_abs[:, 0]
        )


        recon_abs_sc = (
            recon_abs[:, 0]
        )


        xt_abs_sc = (
            xt_abs[:, 0]
        )


        direct_abs_sc = (
            direct_abs[:, 0]
        )


        # ====================================================
        # ADAPTIVE HISTORY
        # ====================================================

        history = getattr(
            diffusion,
            "last_adaptive_history",
            []
        )


        batch_nfe = len(
            history
        )


        total_nfe += (
            batch_nfe * B
        )


        # ====================================================
        # SAVE ADAPTIVE TRAJECTORY
        # ====================================================

        for step_idx, row in enumerate(
            history
        ):

            adaptive_rows.append(
                {
                    "batch": batch_idx,

                    "adaptive_iteration":
                        step_idx + 1,

                    "t":
                        row["t"],

                    "next_t":
                        row["next_t"],

                    "timestep_transition":
                        row["jump"],

                    "R_t":
                        row["residual"],

                    "delta_R_t":
                        row["delta_residual"],
                }
            )


        # ====================================================
        # GLOBAL SLICE INDEX
        # ====================================================
        #
        # sample_count represents the number of slices
        # processed BEFORE this batch.
        # ====================================================

        batch_start_index = (
            sample_count
        )


        # ====================================================
        # PER-SLICE LOOP
        # ====================================================

        batch_results = []


        for i in range(B):

            # ------------------------------------------------
            # Global slice identifier
            # ------------------------------------------------

            global_slice_idx = (
                batch_start_index
                +
                i
            )


            # ------------------------------------------------
            # Per-slice output directory
            # ------------------------------------------------

            batch_dir = os.path.join(
                RESULTS_DIR,
                f"batch_{batch_idx:04d}"
            )


            slice_dir = os.path.join(
                batch_dir,
                f"slice_{i:04d}"
            )


            os.makedirs(
                slice_dir,
                exist_ok=True
            )


            # =================================================
            # GET SLICE IMAGES
            # =================================================

            gt_plot = (
                gt_abs_sc[i]
                .detach()
                .cpu()
                .numpy()
            )


            xt_plot = (
                xt_abs_sc[i]
                .detach()
                .cpu()
                .numpy()
            )


            direct_plot = (
                direct_abs_sc[i]
                .detach()
                .cpu()
                .numpy()
            )


            recon_plot = (
                recon_abs_sc[i]
                .detach()
                .cpu()
                .numpy()
            )


            # =================================================
            # METRICS
            # =================================================

            gt_i = (
                gt_abs_sc[i]
                .unsqueeze(0)
            )


            recon_i = (
                recon_abs_sc[i]
                .unsqueeze(0)
            )


            # -------------------------------------------------
            # NMSE
            # -------------------------------------------------

            nmse = scalar_value(
                calc_nmse_tensor(
                    gt_i,
                    recon_i
                )
            )


            # -------------------------------------------------
            # PSNR
            # -------------------------------------------------

            psnr = scalar_value(
                calc_psnr_tensor(
                    gt_i,
                    recon_i
                )
            )


            # -------------------------------------------------
            # SSIM
            # -------------------------------------------------

            ssim = scalar_value(
                calc_ssim_tensor(
                    gt_i,
                    recon_i
                )
            )


            # =================================================
            # SAVE GROUND TRUTH
            # =================================================

            ground_truth_path = os.path.join(
                slice_dir,
                "ground_truth.png"
            )


            save_image(
                gt_plot,
                "Ground Truth",
                ground_truth_path
            )


            # =================================================
            # SAVE INITIAL DEGRADED IMAGE
            # =================================================

            initial_degraded_path = os.path.join(
                slice_dir,
                "initial_degraded.png"
            )


            save_image(
                xt_plot,
                "Initial KCD Degraded Image",
                initial_degraded_path
            )


            # =================================================
            # SAVE DIRECT RECONSTRUCTION
            # =================================================

            direct_reconstruction_path = os.path.join(
                slice_dir,
                "direct_reconstruction.png"
            )


            save_image(
                direct_plot,
                "Direct Reconstruction",
                direct_reconstruction_path
            )


            # =================================================
            # SAVE RA-PHYKSC-DIFF RECONSTRUCTION
            # =================================================

            adaptive_reconstruction_path = os.path.join(
                slice_dir,
                "adaptive_reconstruction.png"
            )


            save_image(
                recon_plot,
                "RA-PhyKSC-Diff Reconstruction",
                adaptive_reconstruction_path
            )


            # =================================================
            # PER-SLICE CSV RECORD
            # =================================================

            slice_result = {

                # ---------------------------------------------
                # Slice identification
                # ---------------------------------------------

                "batch":
                    batch_idx,

                "sample":
                    i,

                "GlobalSlice":
                    global_slice_idx,


                # ---------------------------------------------
                # Experiment configuration
                # ---------------------------------------------

                "Acceleration":
                    ACCELERATION,

                "CenterFraction":
                    CENTER_FRACTION,

                "Timesteps":
                    TIMESTEPS,


                # ---------------------------------------------
                # Metrics
                # ---------------------------------------------

                "NMSE":
                    nmse,

                "PSNR":
                    psnr,

                "SSIM":
                    ssim,


                # ---------------------------------------------
                # Inference
                # ---------------------------------------------

                "InferenceTime_sec":
                    elapsed / B,

                "NFE":
                    batch_nfe,


                # ---------------------------------------------
                # Physics verification
                # ---------------------------------------------

                "MeasurementError":
                    measurement_error,


                # ---------------------------------------------
                # Visualization paths
                # ---------------------------------------------

                "GroundTruthPath":
                    ground_truth_path,

                "InitialDegradedPath":
                    initial_degraded_path,

                "DirectReconstructionPath":
                    direct_reconstruction_path,

                "AdaptiveReconstructionPath":
                    adaptive_reconstruction_path,
            }


            # -------------------------------------------------
            # Add to global results
            # -------------------------------------------------

            results.append(
                slice_result
            )


            # -------------------------------------------------
            # Add to current batch results
            # -------------------------------------------------

            batch_results.append(
                slice_result
            )


            # -------------------------------------------------
            # Update global sample counter
            # -------------------------------------------------

            sample_count += 1


        # ====================================================
        # SAVE PER-SAMPLE CSV AFTER EVERY BATCH
        # ====================================================
        #
        # This is intentionally inside the batch loop.
        #
        # Therefore, if testing stops after batch N, all
        # results through batch N are already saved.
        # ====================================================

        results_df = pd.DataFrame(
            results
        )


        results_df.to_csv(
            RESULTS_CSV,
            index=False
        )


        # ====================================================
        # SAVE ADAPTIVE TRAJECTORY AFTER EVERY BATCH
        # ====================================================

        adaptive_df = pd.DataFrame(
            adaptive_rows
        )


        adaptive_df.to_csv(
            ADAPTIVE_CSV,
            index=False
        )


        # ====================================================
        # BATCH METRICS
        # ====================================================

        batch_nmse = np.mean(
            [
                x["NMSE"]
                for x in batch_results
            ]
        )


        batch_psnr = np.mean(
            [
                x["PSNR"]
                for x in batch_results
            ]
        )


        batch_ssim = np.mean(
            [
                x["SSIM"]
                for x in batch_results
            ]
        )


        # ====================================================
        # BATCH PROGRESS
        # ====================================================

        print(
            f"\nBatch "
            f"{batch_idx + 1}/"
            f"{len(test_loader)}"
        )


        print(
            f"  Slices processed : "
            f"{len(batch_results)}"
        )


        print(
            f"  Global slices    : "
            f"{batch_start_index} - "
            f"{sample_count - 1}"
        )


        print(
            f"  NMSE             : "
            f"{batch_nmse:.6f}"
        )


        print(
            f"  PSNR             : "
            f"{batch_psnr:.4f}"
        )


        print(
            f"  SSIM             : "
            f"{batch_ssim:.6f}"
        )


        print(
            f"  NFE/sample       : "
            f"{batch_nfe}"
        )


        print(
            f"  Time/sample      : "
            f"{elapsed / B:.4f} sec"
        )


        print(
            f"  Visualizations   : "
            f"{os.path.join(RESULTS_DIR, f'batch_{batch_idx:04d}')}"
        )


        print(
            f"  CSV updated      : "
            f"{RESULTS_CSV}"
        )


# ============================================================
# FINAL RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# FINAL SAVE OF PER-SAMPLE RESULTS
# ============================================================

results_df.to_csv(
    RESULTS_CSV,
    index=False
)


# ============================================================
# FINAL ADAPTIVE TRAJECTORY
# ============================================================

adaptive_df = pd.DataFrame(
    adaptive_rows
)


adaptive_df.to_csv(
    ADAPTIVE_CSV,
    index=False
)


# ============================================================
# ADAPTIVE STEP DISTRIBUTION
# ============================================================

if len(adaptive_df) > 0:

    step_counts = (
        adaptive_df[
            "timestep_transition"
        ]
        .value_counts()
        .sort_index()
    )


    step_distribution = pd.DataFrame(
        {
            "timestep_transition":
                step_counts.index.astype(int),

            "count":
                step_counts.values,

            "fraction":
                (
                    step_counts.values
                    /
                    step_counts.values.sum()
                ),
        }
    )

else:

    step_distribution = pd.DataFrame(
        columns=[
            "timestep_transition",
            "count",
            "fraction"
        ]
    )


step_distribution.to_csv(
    STEP_DISTRIBUTION_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    # --------------------------------------------------------
    # Experiment
    # --------------------------------------------------------

    "Model":
        "RA-PhyKSC-Diff",

    "Acceleration":
        ACCELERATION,

    "CenterFraction":
        CENTER_FRACTION,

    "InitialTimesteps":
        TIMESTEPS,

    "Samples":
        sample_count,


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    "NMSE_mean":
        results_df["NMSE"].mean(),

    "NMSE_std":
        results_df["NMSE"].std(),

    "PSNR_mean":
        results_df["PSNR"].mean(),

    "PSNR_std":
        results_df["PSNR"].std(),

    "SSIM_mean":
        results_df["SSIM"].mean(),

    "SSIM_std":
        results_df["SSIM"].std(),


    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    "MeanInferenceTime_sec_per_sample":
        results_df[
            "InferenceTime_sec"
        ].mean(),

    "TotalInferenceTime_sec":
        total_time,


    # --------------------------------------------------------
    # NFE
    # --------------------------------------------------------

    "TotalNFE":
        total_nfe,

    "MeanNFE_per_sample":
        (
            total_nfe
            /
            max(sample_count, 1)
        ),


    # --------------------------------------------------------
    # Adaptive timestep
    # --------------------------------------------------------

    "MeanTimestepTransition":
        (
            adaptive_df[
                "timestep_transition"
            ].mean()
            if len(adaptive_df) > 0
            else np.nan
        ),


    # --------------------------------------------------------
    # Controller configuration
    # --------------------------------------------------------

    "DeltaR_High":
        DELTA_R_HIGH,

    "DeltaR_Low":
        DELTA_R_LOW,

    "Jump_High":
        JUMP_HIGH,

    "Jump_Medium":
        JUMP_MEDIUM,

    "Jump_Low":
        JUMP_LOW,


    # --------------------------------------------------------
    # Physics verification
    # --------------------------------------------------------

    "MaximumMeasurementError":
        results_df[
            "MeasurementError"
        ].max(),
}


summary_df = pd.DataFrame(
    [summary]
)


summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n"
    +
    "=" * 90
)

print(
    "RA-PhyKSC-Diff TEST RESULTS"
)

print(
    "=" * 90
)


print(
    f"NMSE  : "
    f"{summary['NMSE_mean']:.6f} "
    f"+/- "
    f"{summary['NMSE_std']:.6f}"
)


print(
    f"PSNR  : "
    f"{summary['PSNR_mean']:.4f} "
    f"+/- "
    f"{summary['PSNR_std']:.4f}"
)


print(
    f"SSIM  : "
    f"{summary['SSIM_mean']:.6f} "
    f"+/- "
    f"{summary['SSIM_std']:.6f}"
)


print(
    f"Time  : "
    f"{summary['MeanInferenceTime_sec_per_sample']:.4f} "
    f"sec/sample"
)


print(
    f"NFE   : "
    f"{summary['MeanNFE_per_sample']:.2f} "
    f"evaluations/sample"
)


print(
    f"Max measurement error: "
    f"{summary['MaximumMeasurementError']:.10e}"
)


print(
    "=" * 90
)


print(
    "Total slices processed :",
    sample_count
)


print(
    "Total batches processed:",
    len(test_loader)
)


print(
    "\nPer-sample results     :",
    RESULTS_CSV
)


print(
    "Adaptive trajectory    :",
    ADAPTIVE_CSV
)


print(
    "Step distribution      :",
    STEP_DISTRIBUTION_CSV
)


print(
    "Summary                :",
    SUMMARY_CSV
)


print(
    "Visualizations         :",
    RESULTS_DIR
)


print(
    "=" * 90
)