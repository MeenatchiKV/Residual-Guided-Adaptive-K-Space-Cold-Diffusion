"""Figure 8: fixed vs residual-guided adaptive reverse trajectory (T=1000).

    python figures/make_trajectory_plot.py   ->  figures/fig8_trajectory.{png,pdf}
Uses the RA-PhyKSC-Diff@1000 trajectory of the test slice whose NFE is closest to the median
over all evaluated slices (results/main/trajectories/slice*_T1000.json).
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
res = os.path.join(here, "..", "results", "main")
trajs = {int(os.path.basename(p)[5:8]): json.load(open(p))
         for p in glob.glob(os.path.join(res, "trajectories", "slice*_T1000.json"))}
nfe = pd.Series({i: len(tr) for i, tr in trajs.items()})
idx = int((nfe - nfe.median()).abs().idxmin())
tr = trajs[idx]
meta = pd.read_csv(os.path.join(res, "per_slice.csv")).query("idx == @idx").iloc[0]

it = np.arange(1, len(tr) + 1)
t_next = [h[1] for h in tr]
jumps = np.array([h[0] - h[1] for h in tr])

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(it, 1000 - it, "--", color="tab:blue", lw=2, label=r"PhyKSC-Diff ($\Delta t = 1$)")
ax.plot(it, t_next, "-", color="tab:orange", lw=2, label=r"RA-PhyKSC-Diff ($\Delta t \in \{1, 5, 10\}$)")
ax.set_xlabel("Inference Iteration", fontsize=13)
ax.set_ylabel("Next Diffusion Timestep", fontsize=13)
ax.set_title("Fixed vs. Residual Guided Adaptive Diffusion Trajectory", fontsize=15)
ax.set_xlim(-8, len(tr) + 8)
ax.set_ylim(-40, 1040)
ax.grid(True, ls="--", alpha=0.4)
ax.legend(loc="upper right", fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(here, f"fig8_trajectory.{ext}"), dpi=300 if ext == "png" else None)
print(f"slice idx {idx} ({meta.file} s{meta.slice}): {len(tr)} NFE (median over {len(nfe)} slices "
      f"{nfe.median():.0f}, range {nfe.min()}-{nfe.max()}); fixed path at iteration {len(tr)}: t={1000 - len(tr)}; "
      f"jumps used dt=1 {np.sum(jumps == 1)}, dt=5 {np.sum(jumps == 5)}, dt=10 {np.sum(jumps == 10)}")
