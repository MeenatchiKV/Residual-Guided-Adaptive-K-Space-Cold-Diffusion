"""Build single-coil train/test caches from the fastMRI knee and brain multicoil volumes.

Each slice's 15-coil k-space is reduced to one complex emulated single coil, following the
fastMRI single-coil emulation (Tygert & Zbontar, 2020):
  coil images x_c = IFFT(k_c), centre-cropped to 320x320
  w = argmin_w || sum_c w_c x_c - RSS ||_2     (complex linear least squares)
  emulated coil = sum_c w_c x_c
Being a fixed linear coil combination, the result is a valid complex single-coil acquisition.

Images narrower than 320 (some brain scans) are zero-padded to 320x320 after the centre crop.
Volumes are split 70:30 by volume, stratified by anatomy and contrast.
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np
import torch

from ksc.data import IMG_SIZE, center_crop, ifft2c


def fit_320(x):
    """Centre-crop each of the last two dims to 320, zero-padding any dim that is smaller."""
    h, w = x.shape[-2:]
    top, left = max(0, (h - IMG_SIZE) // 2), max(0, (w - IMG_SIZE) // 2)
    x = x[..., top:top + IMG_SIZE, left:left + IMG_SIZE]
    ph, pw = IMG_SIZE - x.shape[-2], IMG_SIZE - x.shape[-1]
    return torch.nn.functional.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


def virtual_coil(kspace_mc):
    """[Nc, H, W] complex k-space -> ([320, 320] complex emulated single-coil image, rel. error vs RSS)."""
    imgs = fit_320(ifft2c(torch.from_numpy(kspace_mc).cuda()))   # [Nc, 320, 320]
    X = imgs.reshape(imgs.shape[0], -1)
    rss = X.abs().pow(2).sum(0).sqrt()
    w = torch.linalg.lstsq(X.T, rss.to(X.dtype)[:, None]).solution[:, 0]
    vc = X.T @ w
    err = float((vc.abs() - rss).norm() / rss.norm())
    return vc.reshape(IMG_SIZE, IMG_SIZE).cpu().numpy().astype(np.complex64), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=["downloads/knee_multicoil_train:knee:5",
                                                 "downloads/brain_multicoil_train:brain:2"],
                    help="folder:anatomy:num_skip_slice (boundary slices skipped on each side)")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--name", default="fastmri_mc_vc")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default=None,
                    help="reuse a published split file (e.g. cache/fastmri_mc_vc_split.json): only its volumes "
                         "are used, with its train/test assignment, so the full fastMRI folders can be given as --src")
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    given = json.load(open(args.split)) if args.split else None
    wanted = set(given["train"] + given["test"]) if given else None

    vols = {}   # file name -> (path, anatomy, contrast, skip)
    for spec in args.src:
        folder, anatomy, skip = spec.split(":")
        for f in sorted(glob.glob(os.path.join(folder, "*.h5"))):
            if wanted is not None and os.path.basename(f) not in wanted:
                continue
            with h5py.File(f, "r") as h:
                vols[os.path.basename(f)] = (f, anatomy, str(h.attrs["acquisition"]), int(skip))
    if given:
        missing = sorted(wanted - set(vols))
        if missing:
            raise SystemExit(f"{len(missing)} volumes of {args.split} were not found in --src, e.g. {missing[:3]}")
        split = {"train": given["train"], "test": given["test"]}
    else:
        rng = np.random.RandomState(args.seed)
        split = {"train": [], "test": []}
        for stratum in sorted({(v[1], v[2]) for v in vols.values()}):
            group = sorted(k for k, v in vols.items() if (v[1], v[2]) == stratum)
            group = [group[i] for i in rng.permutation(len(group))]
            n_test = int(round(len(group) * args.test_frac))
            split["test"] += sorted(group[:n_test])
            split["train"] += sorted(group[n_test:])
        with open(os.path.join(args.cache_dir, f"{args.name}_split.json"), "w") as fh:
            json.dump({"sources": args.src, "seed": args.seed,
                       "volumes": {k: v[1:3] for k, v in vols.items()}, **split}, fh, indent=1)
    for part, names in split.items():
        print(part, {a: sum(vols[n][1] == a for n in names) for a in ("knee", "brain")})

    for part, names in split.items():
        images, fnames, slices, anat, err = [], [], [], [], []
        for i, name in enumerate(names):
            path, anatomy, _, skip = vols[name]
            with h5py.File(path, "r") as h:
                n = h["kspace"].shape[0]
                for s in range(n):
                    if s < skip or s > n - skip:
                        continue
                    vc, e = virtual_coil(h["kspace"][s])
                    images.append(vc); fnames.append(name); slices.append(s); anat.append(anatomy); err.append(e)
            print(f"{part} [{i + 1}/{len(names)}] {name}  slices so far {len(images)}", flush=True)
        out = os.path.join(args.cache_dir, f"{args.name}_{part}.npz")
        np.savez(out, images=np.stack(images), files=np.array(fnames), slices=np.array(slices),
                 anatomy=np.array(anat), energy=np.array(err))
        anat, err = np.array(anat), np.array(err)
        for a in ("knee", "brain"):
            print(f"{part} {a}: {np.sum(anat == a)} slices; relative error of |emulated| vs RSS "
                  f"mean {err[anat == a].mean():.3f} max {err[anat == a].max():.3f}")


if __name__ == "__main__":
    main()
