"""Figure 7: PSNR-NFE trade-off (fixed PhyKSC-Diff vs RA-PhyKSC-Diff).

    python figures/make_psnr_nfe_plot.py   ->  figures/fig7_psnr_nfe_tradeoff.{png,pdf}
Values come from results/main/table5_ablation.csv (the Table 4 ablation subset).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
t = pd.read_csv(os.path.join(here, "..", "results", "main", "table5_ablation.csv"))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t["Phy NFE"], t["Phy PSNR"], "o-", color="tab:blue", lw=2, ms=7, label="PhyKSC-Diff")
ax.plot(t["RA NFE"], t["RA PSNR"], "s-", color="tab:orange", lw=2, ms=7, label="RA-PhyKSC-Diff")
for T, n, p in zip(t["T"], t["RA NFE"], t["RA PSNR"]):
    ax.annotate(f"T={T}", (n, p), textcoords="offset points", xytext=(7, 5), fontsize=9)
ax.set_xlim(-30, 1050)
ax.set_xlabel("Number of Function Evaluations (NFE)", fontsize=13)
ax.set_ylabel("PSNR (dB)", fontsize=13)
ax.set_title("PSNR–NFE Trade-off", fontsize=15)
ax.grid(True, ls="--", alpha=0.4)
ax.legend(loc="upper right", fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(here, f"fig7_psnr_nfe_tradeoff.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(here, "fig7_psnr_nfe_tradeoff.*"))
