"""Figure 6: qualitative comparison with error maps, zoomed ROIs and per-image metrics.

    python figures/make_qualitative_plot.py   ->  figures/fig6_qualitative.{png,pdf}
Uses results/main/recon/slice*.npz (written by evaluate.py and figures/recon_slice.py).
The ROI is placed automatically on the 64x64 window of the ground truth with the largest
gradient energy (central 70% of the image), so it does not depend on any method's errors.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import uniform_filter

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))
from evaluate import metrics   # noqa: E402  (same NMSE/PSNR/SSIM as the evaluation)

SLICES = [(65, "Brain, T1 post-contrast"), (130, "Brain, T2"), (280, "Knee, PD fat-suppressed")]
COLS = [("gt", "Ground truth"), ("Zero-filled@0", "Zero-filled (R4)"), ("KSC-Diff@1000", "KSC-Diff"),
        ("PhyKSC-Diff@1000", "PhyKSC-Diff"), ("RA-PhyKSC-Diff@1000", "RA-PhyKSC-Diff")]
ROI, ERR_MAX = 64, 0.15


def pick_roi(gt):
    gy, gx = np.gradient(gt)
    energy = uniform_filter(gy ** 2 + gx ** 2, ROI)          # mean energy of each ROI window
    h, w = gt.shape
    m = np.zeros_like(energy, dtype=bool)
    m[int(0.15 * h) + ROI // 2:int(0.85 * h) - ROI // 2, int(0.15 * w) + ROI // 2:int(0.85 * w) - ROI // 2] = True
    cy, cx = np.unravel_index(np.where(m, energy, -1).argmax(), energy.shape)
    return cy - ROI // 2, cx - ROI // 2


fig = plt.figure(figsize=(15, 6.4 * len(SLICES)))
outer = fig.add_gridspec(len(SLICES), 1, hspace=0.12)
for r, (idx, label) in enumerate(SLICES):
    d = np.load(os.path.join(here, "..", "results", "main", "recon", f"slice{idx:03d}.npz"))
    gt = d["gt"]
    vmax = np.percentile(gt, 99.7)
    y0, x0 = pick_roi(gt)
    g = outer[r].subgridspec(3, len(COLS) + 1, height_ratios=[1, 0.42, 1], width_ratios=[1] * len(COLS) + [0.05],
                             hspace=0.04, wspace=0.04)
    for c, (key, title) in enumerate(COLS):
        img = d[key]
        ax = fig.add_subplot(g[0, c])
        ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
        ax.add_patch(Rectangle((x0, y0), ROI, ROI, ec="#ffd21f", fc="none", lw=1.4))
        if r == 0:
            ax.set_title(title, fontsize=13)
        if c == 0:
            ax.set_ylabel(label, fontsize=12)
        if key != "gt":
            nmse, psnr, ssim = metrics(gt, img)
            ax.text(0.03, 0.03, f"{psnr:.2f} dB / {ssim:.4f}", transform=ax.transAxes, color="w",
                    fontsize=10, va="bottom", bbox=dict(fc="black", alpha=0.55, lw=0, pad=2))
        zx = fig.add_subplot(g[1, c])
        zx.imshow(img[y0:y0 + ROI, x0:x0 + ROI], cmap="gray", vmin=0, vmax=vmax)
        for s in zx.spines.values():
            s.set_edgecolor("#ffd21f"); s.set_linewidth(1.4)
        ex = fig.add_subplot(g[2, c])
        if key == "gt":
            ex.axis("off")
            ex.text(0.5, 0.5, "|error| / max(GT)\nshared scale\n0 – %.2f" % ERR_MAX, ha="center",
                    va="center", fontsize=11, transform=ex.transAxes)
        else:
            im = ex.imshow(np.abs(img - gt) / gt.max(), cmap="inferno", vmin=0, vmax=ERR_MAX)
        for a in (ax, zx, ex):
            a.set_xticks([]); a.set_yticks([])
    cb = fig.colorbar(im, cax=fig.add_subplot(g[2, -1]))
    cb.ax.tick_params(labelsize=9)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(here, f"fig6_qualitative.{ext}"), dpi=200 if ext == "png" else None,
                bbox_inches="tight")
print("saved", os.path.join(here, "fig6_qualitative.*"))
