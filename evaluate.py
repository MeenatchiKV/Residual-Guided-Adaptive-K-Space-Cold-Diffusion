"""Evaluation for RQ1-RQ3 on the R4 test split, one slice at a time (batch 1, timed).

Every test slice (Tables 1-3, RQ1):
  Zero-filled
  KSC-Diff        @1000   KSC checkpoint, no DC
  PhyKSC-Diff     @1000   PhyKSC checkpoint (DC-trained), DC layer kept in the forward pass
  RA-PhyKSC-Diff  @1000   PhyKSC checkpoint + residual-adaptive dt

Ablation subset (--ablation-per-volume evenly spaced slices per test volume; RQ2, RQ3):
  PhyKSC-Diff / RA-PhyKSC-Diff @ T in {20, 50, 100, 200, 500}
  Uniform-matched @1000   fixed dt, NFE = that slice's RA-PhyKSC-Diff@1000 NFE
  Random-order    @1000   RA's own jump sizes for that slice, in shuffled order
  RA-PhyKSC-Diff (thr x s) @1000   controller thresholds scaled by s

Probe subset (--probe-per-volume slices per volume; RQ2): an untimed PhyKSC-Diff@1000 pass that
records R_t, dR_t and the deviation a dt=5 / dt=10 jump would have caused at every state.

Rows go to <out>/per_slice.csv as they finish; rerunning resumes.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from skimage.metrics import structural_similarity

from ksc.data import KneeSinglecoil, ifft2c
from ksc.diffusion import KSpaceColdDiffusion
from ksc.unet import Unet

FIELDS = ["idx", "file", "slice", "method", "T", "nfe", "time_s", "nmse", "psnr", "ssim"]
THR_SCALES = [0.5, 2.0, 4.0]


def metrics(gt, pred):
    gt, pred = gt.astype(np.float64), pred.astype(np.float64)
    nmse = np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2
    mse = np.mean((gt - pred) ** 2)
    psnr = 10 * np.log10(gt.max() ** 2 / mse)
    ssim = structural_similarity(gt, pred, data_range=gt.max())
    return nmse, psnr, ssim


def load(ckpt, dev):
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=2).to(dev).eval()
    ck = torch.load(ckpt, map_location=dev)
    model.load_state_dict(ck["ema"])
    print(f"{ckpt}: epoch {ck.get('epoch')} step {ck.get('step')} "
          f"dc_train={ck.get('args', {}).get('dc_train', False)}")
    return KSpaceColdDiffusion(model, 1000)


def per_volume_subset(files, n_per_volume):
    """Indices of n evenly spaced slices from every volume."""
    chosen = []
    for fname in np.unique(files):
        idxs = np.where(files == fname)[0]
        pick = np.linspace(0, len(idxs) - 1, min(n_per_volume, len(idxs))).round().astype(int)
        chosen += [int(i) for i in idxs[pick]]
    return sorted(set(chosen))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ksc-ckpt", default="runs/ksc/last.pt")
    p.add_argument("--phy-ckpt", default="runs/phyksc/last.pt")
    p.add_argument("--data", default="cache/fastmri_mc_vc_test.npz", help="cache built by preprocess.py")
    p.add_argument("--out", default="results/main")
    p.add_argument("--main-T", type=int, default=1000, help="diffusion configuration of the main tables")
    p.add_argument("--T-list", type=int, nargs="+", default=[20, 50, 100, 200, 500])
    p.add_argument("--ablation-per-volume", type=int, default=5)
    p.add_argument("--probe-per-volume", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N slices")
    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    for sub in ("", "recon", "trajectories"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    ds = KneeSinglecoil(args.data, acceleration=4, center_fraction=0.08, fixed_mask=True, seed=1234)
    ksc, phy = load(args.ksc_ckpt, dev), load(args.phy_ckpt, dev)
    n = args.limit or len(ds)
    ablation = set(per_volume_subset(ds.files, args.ablation_per_volume))
    probe_set = set(per_volume_subset(ds.files, args.probe_per_volume))
    anatomy = np.where(np.char.startswith(ds.files.astype(str), "file_brain"), "brain", "knee")
    save_recon = set()
    for a in ("knee", "brain"):
        cand = [j for j in sorted(ablation) if anatomy[j] == a]
        save_recon.update(cand[len(cand) // 3::len(cand) // 3][:2])
    print(f"slices {n} | ablation subset {len([i for i in ablation if i < n])} | "
          f"probe subset {len([i for i in probe_set if i < n])} | recon saved for {sorted(save_recon)}")

    csv_path = os.path.join(args.out, "per_slice.csv")
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as fh:
            done = {(int(r["idx"]), r["method"], int(r["T"])) for r in csv.DictReader(fh)}
    f = open(csv_path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if not done:
        w.writeheader()
    probe_path = os.path.join(args.out, "probe.csv")
    probe_done = set()
    if os.path.exists(probe_path):
        with open(probe_path) as fh:
            probe_done = {int(r["idx"]) for r in csv.DictReader(fh)}
    pf = open(probe_path, "a", newline="")
    pw = csv.writer(pf)
    if not probe_done:
        pw.writerow(["idx", "t", "R", "dR", "dev5", "dev10"])

    k, mask, order, _ = ds[0]   # warm-up so cudnn autotuning is not charged to the first slice
    phy.sample(k[None].to(dev), mask[None].to(dev), order[None].to(dev), steps=5)

    for idx in range(n):
        k, mask, order, _ = ds[idx]
        k, mask, order = k[None].to(dev), mask[None].to(dev), order[None].to(dev)
        gt = ifft2c(k)[0].abs().cpu().numpy()
        base = {"idx": idx, "file": ds.files[idx], "slice": int(ds.slices[idx])}
        recons = {}

        def run(name, T, diff, steps=None, **kw):
            """Run one configuration (skipped if already in the CSV); returns its history."""
            if (idx, name, T) in done:
                return None
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            img, nfe, hist, _ = diff.sample(k, mask, order, steps=steps or T, **kw)
            torch.cuda.synchronize()
            secs = time.perf_counter() - t0
            pred = img[0].abs().cpu().numpy()
            nmse, psnr, ssim = metrics(gt, pred)
            w.writerow({**base, "method": name, "T": T, "nfe": int(nfe[0]), "time_s": round(secs, 4),
                        "nmse": nmse, "psnr": psnr, "ssim": ssim})
            f.flush()
            recons[f"{name}@{T}"] = pred
            return hist

        def traj_file(T):
            return os.path.join(args.out, "trajectories", f"slice{idx:03d}_T{T}.json")

        def run_ra(T, **kw):
            hist = run("RA-PhyKSC-Diff", T, phy, physics=True, adaptive=True, record=True, **kw)
            if hist is not None:
                with open(traj_file(T), "w") as jf:
                    json.dump(hist[0], jf)

        if (idx, "Zero-filled", 0) not in done:
            zf = ifft2c(k * mask[:, None, :])[0].abs().cpu().numpy()
            w.writerow({**base, "method": "Zero-filled", "T": 0, "nfe": 0, "time_s": 0,
                        **dict(zip(["nmse", "psnr", "ssim"], metrics(gt, zf)))})
            recons["Zero-filled@0"] = zf

        # ---- every slice: RQ1 and the headline RQ3 comparison
        MT = args.main_T
        run("KSC-Diff", MT, ksc, physics=False)
        run("PhyKSC-Diff", MT, phy, physics=True)
        run_ra(MT)

        # ---- ablation subset: diffusion-time ablation, matched-NFE controls, threshold sweep
        if idx in ablation:
            for T in args.T_list:
                run("PhyKSC-Diff", T, phy, physics=True)
                run_ra(T)
            grid = 1000 / MT
            incs = [round((a - b) / grid) for a, b, _, _ in json.load(open(traj_file(MT)))]   # RA jumps
            # fixed dt on a uniform grid with the same NFE as RA on this slice
            run("Uniform-matched", MT, phy, steps=len(incs), physics=True)
            # RA's own jump sizes in random order: same NFE, no residual information
            shuffled = [int(s) for s in np.random.RandomState(idx).permutation(incs)]
            run("Random-order", MT, phy, physics=True, schedule=shuffled)
            for s in THR_SCALES:
                run(f"RA-PhyKSC-Diff (thr x{s:g})", MT, phy, physics=True, adaptive=True,
                    thr_high=0.02 * s, thr_low=0.005 * s)

        # ---- probe subset: does dR_t predict the error of skipping? (separate, untimed pass)
        if idx in probe_set and idx not in probe_done:
            _, _, _, rows = phy.sample(k, mask, order, steps=MT, physics=True, probe_jumps=(5, 10))
            for t, R, dR, devs in rows:
                pw.writerow([idx, t, R, dR, devs.get(5), devs.get(10)])
            pf.flush()

        if idx in save_recon and len(recons) >= 4:
            np.savez_compressed(os.path.join(args.out, "recon", f"slice{idx:03d}.npz"), gt=gt,
                                **{kk.replace(" ", "_"): v for kk, v in recons.items()})
        print(f"[{idx + 1}/{n}] {ds.files[idx]} s{int(ds.slices[idx])} done", flush=True)
    f.close()
    pf.close()


if __name__ == "__main__":
    main()
