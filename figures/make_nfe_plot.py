"""Figure 6: inference NFE vs diffusion timestep (fixed PhyKSC-Diff vs RA-PhyKSC-Diff).

    python figures/make_nfe_plot.py   ->  figures/fig6_nfe_vs_timestep.{png,pdf}
Values come from results/main/table5_ablation.csv (the Table 4 ablation subset).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
t = pd.read_csv(os.path.join(here, "..", "results", "main", "table5_ablation.csv"))
x = range(len(t))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, t["Phy NFE"], "o-", color="tab:blue", lw=2, ms=7, label="PhyKSC-Diffusion")
ax.plot(x, t["RA NFE"], "s-", color="tab:orange", lw=2, ms=7, label="RA-PhyKSC-Diffusion")
for i, n in zip(x, t["RA NFE"]):
    below = n < 60   # small values: label under the point so it does not sit on the blue curve
    ax.annotate(f"{n:.2f}", (i, n), textcoords="offset points", xytext=(0, -17 if below else 9),
                ha="center", fontsize=10)
ax.set_xticks(list(x), [str(v) for v in t["T"]])
ax.set_ylim(-60, 1060)
ax.set_xlabel("Diffusion Timestep", fontsize=13)
ax.set_ylabel("Number of Function Evaluations (NFE)", fontsize=13)
ax.set_title("Inference NFE vs Diffusion Timestep", fontsize=15)
ax.grid(True, ls="--", alpha=0.4)
ax.legend(loc="upper left", fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(here, f"fig6_nfe_vs_timestep.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(here, "fig6_nfe_vs_timestep.*"))
