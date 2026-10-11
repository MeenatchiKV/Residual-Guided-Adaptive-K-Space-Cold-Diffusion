# ============================================================
# train.py
# Physics-Guided K-Space Cold Diffusion
# Residual-Adaptive version (RA-PhyKSC-Diff)
# Single-Coil fastMRI Knee
#
# IMPORTANT:
# The residual-adaptive controller is an INFERENCE-TIME module.
# Therefore the network is trained with the same KCD objective as
# Physics-KCD. Adaptive timestep selection is activated in test.py.
# ============================================================

import os
import random
import pathlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------
from utils.mri_data import SliceDataset
from utils.data_transform import DataTransform_PhysicsDiffusion
from utils.sample_mask import RandomMaskDiffusion
from net.u_net_diffusion import Unet
from physics_kspace_diffusion import PhysicsKspaceDiffusion


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MRI
IMAGE_SIZE = 320
ACCELERATION = 6
CENTER_FRACTION = 0.03

# Dataset
TRAIN_PATH = (
    r"C:\Users\meena\Documents\Amrita"
    r"\MRIReconstruction\datasets\multi-single\train"
)

VAL_PATH = (
    r"C:\Users\meena\Documents\Amrita"
    r"\MRIReconstruction\datasets\multi-single\test"
)

# Training
BATCH_SIZE = 6
NUM_EPOCHS = 1                 # change to 20/50 for full training
LEARNING_RATE = 2e-5
GRADIENT_ACCUMULATION = 2
NUM_WORKERS = 0

# Diffusion
TIMESTEPS = 1000
LOSS_TYPE = "l1"
CH_MID = 64

# EMA
EMA_DECAY = 0.995
EMA_START_STEP = 2000

# Output
OUTPUT_ROOT = "./saved_models/fastmri_knee"
EXPERIMENT_NAME = (
    f"ra_physics_kcd_fastmri_{ACCELERATION}x_T{TIMESTEPS}"
)
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, EXPERIMENT_NAME)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducibility. Set False if maximum speed is preferred.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# EMA
# ============================================================

@torch.no_grad()
def update_ema(ema_model, model, decay):
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())

    for name in ema_params.keys():
        ema_params[name].mul_(decay).add_(model_params[name], alpha=1.0 - decay)

    # Keep buffers synchronized.
    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name in ema_buffers.keys():
        ema_buffers[name].copy_(model_buffers[name])


# ============================================================
# DATA
# ============================================================

print("=" * 80)
print("RA-PhyKSC-Diff TRAINING")
print("=" * 80)
print("PyTorch       :", torch.__version__)
print("Device        :", DEVICE)

if torch.cuda.is_available():
    print("GPU           :", torch.cuda.get_device_name(0))
    print(
        "GPU memory    :",
        round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
            2,
        ),
        "GB",
    )

print("Image size    :", IMAGE_SIZE)
print("Acceleration  : R" + str(ACCELERATION))
print("Center frac.  :", CENTER_FRACTION)
print("Batch size    :", BATCH_SIZE)
print("Epochs        :", NUM_EPOCHS)
print("Timesteps     :", TIMESTEPS)
print("Learning rate :", LEARNING_RATE)
print("Output        :", OUTPUT_DIR)
print("=" * 80)


mask_func = RandomMaskDiffusion(
    acceleration=ACCELERATION,
    center_fraction=CENTER_FRACTION,
    size=(1, IMAGE_SIZE, IMAGE_SIZE),
)

data_transform = DataTransform_PhysicsDiffusion(
    mask_func,
    img_size=IMAGE_SIZE,
    flag_singlecoil=True,
)

train_dataset = SliceDataset(
    root=pathlib.Path(TRAIN_PATH),
    transform=data_transform,
    challenge="singlecoil",
    num_skip_slice=5,
)

val_dataset = SliceDataset(
    root=pathlib.Path(VAL_PATH),
    transform=data_transform,
    challenge="singlecoil",
    num_skip_slice=5,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

print("Training slices:", len(train_dataset))
print("Validation slices:", len(val_dataset))
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# ============================================================
# MODEL
# ============================================================

model = Unet(
    dim=CH_MID,
    dim_mults=(1, 2, 4, 8),
    channels=2,
).to(DEVICE)

# Same architecture is copied for EMA evaluation.
ema_model = Unet(
    dim=CH_MID,
    dim_mults=(1, 2, 4, 8),
    channels=2,
).to(DEVICE)
ema_model.load_state_dict(model.state_dict())


diffusion = PhysicsKspaceDiffusion(
    model,
    image_size=IMAGE_SIZE,
    device_of_kernel=DEVICE,
    channels=2,
    timesteps=TIMESTEPS,
    loss_type=LOSS_TYPE,
    blur_routine="Constant",
    train_routine="Final",
    sampling_routine="x0_step_down",
    discrete=False,
    dc_weight=1.0,
    debug=False,
    adaptive_residual=False,   # training does not use adaptive sampling
).to(DEVICE)

# EMA diffusion wrapper. It is not trained; only its denoiser weights are
# updated from ema_model after each optimizer update.
ema_diffusion = PhysicsKspaceDiffusion(
    ema_model,
    image_size=IMAGE_SIZE,
    device_of_kernel=DEVICE,
    channels=2,
    timesteps=TIMESTEPS,
    loss_type=LOSS_TYPE,
    blur_routine="Constant",
    train_routine="Final",
    sampling_routine="x0_step_down",
    discrete=False,
    dc_weight=1.0,
    debug=False,
    adaptive_residual=False,
).to(DEVICE)


optimizer = torch.optim.Adam(
    diffusion.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def save_checkpoint(step, epoch, loss, filename):
    path = os.path.join(OUTPUT_DIR, filename)

    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "model": diffusion.state_dict(),
            "ema": ema_diffusion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {
                "image_size": IMAGE_SIZE,
                "acceleration": ACCELERATION,
                "center_fraction": CENTER_FRACTION,
                "timesteps": TIMESTEPS,
                "loss_type": LOSS_TYPE,
                "ch_mid": CH_MID,
                "adaptive_residual": False,
                "delta_residual_high": 0.02,
                "delta_residual_low": 0.005,
                "adaptive_jump_high": 1,
                "adaptive_jump_medium": 5,
                "adaptive_jump_low": 10,
            },
        },
        path,
    )
    print("Checkpoint saved:", path)
    return path


# ============================================================
# TRAINING LOOP
# ============================================================

history = []
global_step = 0
best_loss = float("inf")

print("\nStarting training...")

for epoch in range(1, NUM_EPOCHS + 1):

    diffusion.train()
    running_loss = 0.0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{NUM_EPOCHS}",
    )

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(pbar):

        # ----------------------------------------------------
        # Support both transform formats used in the project:
        #   3 outputs: kspace, mask, mask_fold
        #   4 outputs: kspace, measured_kspace, mask, mask_fold
        # ----------------------------------------------------
        if len(batch) == 4:
            kspace, _, mask, mask_fold = batch
        elif len(batch) == 3:
            kspace, mask, mask_fold = batch
        else:
            raise RuntimeError(
                f"Unexpected batch length: {len(batch)}. "
                "Expected 3 or 4 outputs."
            )

        kspace = kspace.to(DEVICE, non_blocking=True).float()
        mask = mask.to(DEVICE, non_blocking=True).float()
        mask_fold = mask_fold.to(DEVICE, non_blocking=True).float()

        loss = diffusion(
            kspace,
            mask,
            mask_fold,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch={epoch}, batch={batch_idx}: {loss.item()}"
            )

        scaled_loss = loss / GRADIENT_ACCUMULATION
        scaled_loss.backward()

        running_loss += loss.item()

        if (
            (batch_idx + 1) % GRADIENT_ACCUMULATION == 0
            or (batch_idx + 1) == len(train_loader)
        ):
            torch.nn.utils.clip_grad_norm_(
                diffusion.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if global_step < EMA_START_STEP:
                ema_diffusion.load_state_dict(
                    diffusion.state_dict()
                )
            else:
                update_ema(
                    ema_diffusion,
                    diffusion,
                    EMA_DECAY,
                )

        current_avg = running_loss / (batch_idx + 1)
        pbar.set_postfix(loss=f"{current_avg:.6f}")

        history.append(
            {
                "epoch": epoch,
                "batch": batch_idx + 1,
                "global_step": global_step,
                "loss": loss.item(),
            }
        )

    epoch_loss = running_loss / len(train_loader)

    print(
        f"Epoch {epoch}/{NUM_EPOCHS} | "
        f"Average loss: {epoch_loss:.6f}"
    )

    # Save latest checkpoint every epoch.
    save_checkpoint(
        global_step,
        epoch,
        epoch_loss,
        f"model_epoch_{epoch}.pt",
    )

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        save_checkpoint(
            global_step,
            epoch,
            epoch_loss,
            "best.pt",
        )


# ============================================================
# FINAL CHECKPOINT
# ============================================================

final_path = save_checkpoint(
    global_step,
    NUM_EPOCHS,
    best_loss,
    "model_final.pt",
)


# ============================================================
# SAVE LOSS CSV
# ============================================================

loss_df = pd.DataFrame(history)
loss_csv = os.path.join(OUTPUT_DIR, "training_losses.csv")
loss_df.to_csv(loss_csv, index=False)

print("Training losses saved:", loss_csv)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config_df = pd.DataFrame(
    [
        {
            "Model": "RA-PhyKSC-Diff",
            "Acceleration": ACCELERATION,
            "CenterFraction": CENTER_FRACTION,
            "ImageSize": IMAGE_SIZE,
            "BatchSize": BATCH_SIZE,
            "Epochs": NUM_EPOCHS,
            "LearningRate": LEARNING_RATE,
            "Timesteps": TIMESTEPS,
            "Loss": LOSS_TYPE,
            "AdaptiveAtTraining": False,
            "DeltaR_High": 0.02,
            "DeltaR_Low": 0.005,
            "Jump_HighResidualChange": 1,
            "Jump_MediumResidualChange": 5,
            "Jump_LowResidualChange": 10,
        }
    ]
)

config_csv = os.path.join(OUTPUT_DIR, "experiment_config.csv")
config_df.to_csv(config_csv, index=False)


print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print("Best loss       :", best_loss)
print("Optimizer steps :", global_step)
print("Final checkpoint:", final_path)
print("Loss CSV        :", loss_csv)
print("Config CSV      :", config_csv)
print("=" * 80)
