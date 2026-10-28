"""Figure 5: average NMSE / PSNR / SSIM of KSC-Diff and PhyKSC-Diff (T = 1000, all test slices).

    python figures/make_fig5_bars.py   ->  figures/fig5_rq1_bars.{png,pdf}
Values come from results/main/per_slice.csv.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(here, "..", "results", "main", "per_slice.csv"))
d = d[d["T"] == 1000]
m = d.groupby("method")[["nmse", "psnr", "ssim"]].mean()
n = d[d.method == "PhyKSC-Diff"].idx.nunique()

panels = [("nmse", "Average NMSE", "{:.4f}", (0, None)), ("psnr", "Average PSNR (dB)", "{:.2f}", (28.8, 30.0)),
          ("ssim", "Average SSIM", "{:.4f}", (0.700, 0.715))]
fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, (col, title, f, ylim) in zip(axs, panels):
    vals = [m.loc["KSC-Diff", col], m.loc["PhyKSC-Diff", col]]
    bars = ax.bar(["KSC-Diff", "PhyKSC-Diff"], vals, color=["#2161b3", "#3d9e3d"], width=0.6)
    for b, v in zip(bars, vals):
        ax.annotate(f.format(v), (b.get_x() + b.get_width() / 2, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(ylim[0], ylim[1] if ylim[1] else max(vals) * 1.15)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(here, f"fig5_rq1_bars.{ext}"), dpi=300 if ext == "png" else None)
print(f"saved fig5_rq1_bars ({n} slices):", m.loc[["KSC-Diff", "PhyKSC-Diff"]].round(5).to_dict("index"))
