"""fastMRI knee single-coil data pipeline.

Slices come from a cache .npz of 320x320 complex images: either built by preprocess.py from the
multicoil volumes (virtual coil), or by singlecoil_cache() from fastMRI single-coil files. Each slice is converted to a 320x320 complex image (centre crop of the IFFT of the raw
k-space), re-transformed to k-space, and scaled by 1 / max|zero-filled image| as in the
original KSC-Diff implementation (Shen et al., 2024).
"""
import glob
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_SIZE = 320


def fft2c(x):
    """Centred orthonormal 2-D FFT over the last two dims of a complex tensor."""
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


def ifft2c(k):
    k = torch.fft.ifftshift(k, dim=(-2, -1))
    k = torch.fft.ifft2(k, norm="ortho")
    return torch.fft.fftshift(k, dim=(-2, -1))


def center_crop(x, size):
    h, w = x.shape[-2:]
    top, left = (h - size) // 2, (w - size) // 2
    return x[..., top:top + size, left:left + size]


def random_column_mask(width, acceleration, center_fraction, rng):
    """Cartesian random column mask (fastMRI RandomMaskFunc convention).

    Returns a float32 vector of length `width`, 1 = acquired column.
    """
    num_low = int(round(width * center_fraction))
    prob = (width / acceleration - num_low) / (width - num_low)
    pad = (width - num_low + 1) // 2
    candidates = np.concatenate([np.arange(0, pad), np.arange(pad + num_low, width)])
    perm = rng.permutation(len(candidates))
    num_masked = int(round((1 - prob) * len(candidates)))
    mask = np.ones(width, dtype=np.float32)
    mask[candidates[perm[:num_masked]]] = 0.0
    return mask


def build_cache(root, cache_path, num_skip_slice=5):
    """Pre-compute normalised complex k-space for every usable slice of a split."""
    files = sorted(glob.glob(os.path.join(root, "*.h5")))
    images, meta = [], []
    for f in files:
        with h5py.File(f, "r") as h:
            kspace = h["kspace"][()]  # [S, H, W] complex64
        n = kspace.shape[0]
        for s in range(n):
            # Same boundary-slice rule as the original SliceDataset(num_skip_slice=5)
            if s < num_skip_slice or s > n - num_skip_slice:
                continue
            img = ifft2c(torch.from_numpy(kspace[s]))
            img = center_crop(img, IMG_SIZE)
            images.append(img.numpy().astype(np.complex64))
            meta.append((os.path.basename(f), s))
    np.savez(cache_path, images=np.stack(images),
             files=np.array([m[0] for m in meta]), slices=np.array([m[1] for m in meta]))
    return cache_path


def singlecoil_cache(root, cache_dir="cache", num_skip_slice=5):
    """Cache path for a folder of fastMRI single-coil .h5 files (built on first use)."""
    os.makedirs(cache_dir, exist_ok=True)
    name = os.path.basename(os.path.normpath(root))
    cache = os.path.join(cache_dir, f"{name}_skip{num_skip_slice}.npz")
    if not os.path.exists(cache):
        build_cache(root, cache, num_skip_slice)
    return cache


class KneeSinglecoil(Dataset):
    """Returns (normalised full k-space [H,W] complex, column mask [W], KCD column order [W]).

    `order[c]` is the rank at which unsampled column c is removed by the cold-diffusion
    degradation (sampled columns get rank = +inf, i.e. are never removed). For the test
    split, mask and order are seeded per slice so every method sees identical inputs.
    """

    def __init__(self, cache_path, acceleration=4, center_fraction=0.08, fixed_mask=False, seed=0):
        d = np.load(cache_path)
        self.images = d["images"]
        self.files = d["files"]
        self.slices = d["slices"]
        self.acc = acceleration
        self.cf = center_fraction
        self.fixed_mask = fixed_mask
        self.seed = seed

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.fixed_mask:
            rng = np.random.RandomState(self.seed + idx)
        else:
            rng = np.random.RandomState()
        mask = random_column_mask(IMG_SIZE, self.acc, self.cf, rng)
        order = np.full(IMG_SIZE, np.inf, dtype=np.float32)
        unsampled = np.where(mask == 0)[0]
        order[unsampled[rng.permutation(len(unsampled))]] = np.arange(len(unsampled))

        img = torch.from_numpy(self.images[idx])
        k = fft2c(img)
        zf = ifft2c(k * torch.from_numpy(mask))
        k = k / zf.abs().max()
        return k, torch.from_numpy(mask), torch.from_numpy(order), idx
