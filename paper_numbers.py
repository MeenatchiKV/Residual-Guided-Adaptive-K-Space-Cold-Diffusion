"""All numbers reported in the manuscript, computed from results/main (full test set).

    python paper_numbers.py [results_dir]  ->  <results_dir>/paper_numbers.json  (default results/main)
Experiment I and II two-sided tests use Holm correction across their four metrics
(pooled analysis, no per-anatomy split), as described in Section 4.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

from report import holm, noninferiority

RES = sys.argv[1] if len(sys.argv) > 1 else "results/main"
MARGINS = {"psnr": 0.1, "ssim": 0.005, "nmse": 0.02}
d = pd.read_csv(os.path.join(RES, "per_slice.csv"))
d["key"] = d.method + "@" + d["T"].astype(str)
P = d.pivot_table(index="idx", columns="key", values=["nmse", "psnr", "ssim", "nfe", "time_s"])
out = {}


def two_sided(a, b):
    rows = []
    for m in ["nmse", "psnr", "ssim", "time_s"]:
        x, y = P[m][a].dropna(), P[m][b].dropna()
        x, y = x.align(y, join="inner")
        dd = y - x
        r = wilcoxon(x, y)
        better = (dd < 0).mean() if m in ("nmse", "time_s") else (dd > 0).mean()
        rows.append(dict(metric=m, median_a=x.median(), median_b=y.median(), median_paired=dd.median(),
                         W=float(r.statistic), p=float(r.pvalue), share_b_better=float(better), n=len(dd)))
    for r, q in zip(rows, holm([r["p"] for r in rows])):
        r["holm"] = float(q)
    return rows


def means(key, idx=None):
    s = P.xs(key, axis=1, level=1)
    if idx is not None:
        s = s.loc[idx]
    return {c: float(s[c].mean()) for c in ["nmse", "psnr", "ssim", "nfe", "time_s"]}


# ---------------- Experiment I (all 591 slices)
K, PH, RA = "KSC-Diff@1000", "PhyKSC-Diff@1000", "RA-PhyKSC-Diff@1000"
out["n_main"] = int(P["psnr"][PH].notna().sum())
out["exp1_means"] = {"zero_filled": means("Zero-filled@0"), "ksc": means(K), "phy": means(PH)}
k, p = out["exp1_means"]["ksc"], out["exp1_means"]["phy"]
out["exp1_improvement"] = {"nmse_rel_pct": (k["nmse"] - p["nmse"]) / k["nmse"] * 100,
                           "psnr_db": p["psnr"] - k["psnr"], "ssim": p["ssim"] - k["ssim"]}
out["exp1_tests"] = two_sided(K, PH)

# ---------------- Experiment II at T=1000 (all slices)
out["exp2_means"] = {"phy": means(PH), "ra": means(RA)}
out["exp2_tests"] = two_sided(PH, RA)
ra_nfe = P["nfe"][RA]
out["ra_nfe"] = {"mean": float(ra_nfe.mean()), "median": float(ra_nfe.median()),
                 "min": int(ra_nfe.min()), "max": int(ra_nfe.max())}
out["exp2_max_abs_dpsnr"] = float((P["psnr"][RA] - P["psnr"][PH]).abs().max())

# ---------------- non-inferiority (RQ3)
abl_idx = P["psnr"]["Random-order@1000"].dropna().index
out["n_ablation"] = len(abl_idx)
sub = d[d.idx.isin(abl_idx)]
ni = [noninferiority(d[d.idx.isin(P.index)], RA, MARGINS, 0.05)]
ni[0]["Method"] += " (all slices)"
for key in ([f"PhyKSC-Diff@{T}" for T in (20, 50, 100, 200, 500)] +
            [f"RA-PhyKSC-Diff@{T}" for T in (20, 50, 100, 200, 500, 1000)] +
            ["Uniform-matched@1000", "Random-order@1000"] +
            [f"RA-PhyKSC-Diff (thr x{s})@1000" for s in ("0.5", "2", "4")]):
    ni.append(noninferiority(sub, key, MARGINS, 0.05))
out["noninferiority"] = [{k2: (v.item() if hasattr(v, "item") else v) for k2, v in r.items()} for r in ni]

# ---------------- ablation table (subset)
out["ablation"] = {T: {"phy": means(f"PhyKSC-Diff@{T}", abl_idx), "ra": means(f"RA-PhyKSC-Diff@{T}", abl_idx)}
                   for T in (20, 50, 100, 200, 500, 1000)}

# ---------------- Experiment III (RQ2)
out["controls"] = {n: means(k2, abl_idx) for n, k2 in
                   [("ra", RA), ("uniform", "Uniform-matched@1000"), ("random", "Random-order@1000"), ("phy", PH)]}
ctrl = []
for name, key in [("uniform", "Uniform-matched@1000"), ("random", "Random-order@1000")]:
    for m in ["nmse", "psnr", "ssim"]:
        x, y = P[m][key].loc[abl_idx], P[m][RA].loc[abl_idx]
        r = wilcoxon(x, y)
        ctrl.append(dict(control=name, metric=m, median_paired=float((y - x).median()), W=float(r.statistic),
                         p=float(r.pvalue)))
for r, q in zip(ctrl, holm([r["p"] for r in ctrl])):
    r["holm"] = float(q)
out["controls_tests"] = ctrl
out["thresholds"] = {s: means(f"RA-PhyKSC-Diff (thr x{s})@1000" if s != "1" else RA, abl_idx)
                     for s in ("0.5", "1", "2", "4")}

pr = pd.read_csv(os.path.join(RES, "probe.csv")).dropna(subset=["dR"])
out["probe"] = {"slices": int(pr.idx.nunique())}
for j in (5, 10):
    q = pr.dropna(subset=[f"dev{j}"])
    rho = spearmanr(q.dR, q[f"dev{j}"])
    out["probe"][f"dt{j}"] = {"states": len(q), "rho": float(rho.statistic), "p": float(rho.pvalue),
                              "auc": float(roc_auc_score(q[f"dev{j}"] > q[f"dev{j}"].median(), q.dR))}
regions = pd.cut(pr.dR, [-np.inf, 0.005, 0.02, np.inf], labels=["dt10", "dt5", "dt1"])
out["probe"]["regions"] = {str(r): {"states": int(len(g)), "mean_dev5_pct": float(g.dev5.mean() * 100),
                                    "mean_dev10_pct": float(g.dev10.mean() * 100)}
                           for r, g in pr.groupby(regions, observed=True)}

# ---------------- trajectory figure slice (median NFE)
trajs = {int(os.path.basename(f)[5:8]): json.load(open(f)) for f in glob.glob(os.path.join(RES, "trajectories", "slice*_T1000.json"))}
nfe = pd.Series({i: len(t) for i, t in trajs.items()})
mid = int((nfe - nfe.median()).abs().idxmin())
jumps = np.array([h[0] - h[1] for h in trajs[mid]])
out["trajectory_slice"] = {"idx": mid, "nfe": int(len(jumps)), "fixed_t_at_same_iter": int(1000 - len(jumps)),
                           "dt1": int((jumps == 1).sum()), "dt5": int((jumps == 5).sum()), "dt10": int((jumps == 10).sum())}

with open(os.path.join(RES, "paper_numbers.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
