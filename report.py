"""Tables, statistics and figures for RQ1-RQ3 from <res>/per_slice.csv and <res>/probe.csv.

RQ1  KSC-Diff vs PhyKSC-Diff (DC in training), all slices, two-sided Wilcoxon + Holm.
RQ2  Does the residual signal pick good moments to jump?
       - matched-NFE controls: RA vs Uniform-matched vs Random-order (same NFE)
       - probe: Spearman / AUC between dR_t and the deviation a dt=10 (dt=5) jump would cause
       - controller threshold sensitivity
RQ3  Non-inferiority of each accelerated sampler vs PhyKSC-Diff@1000 with predefined margins
     (one-sided Wilcoxon per metric; equivalence claimed only if all metrics pass:
     intersection-union test, each at alpha).
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

METRICS = [("nmse", "NMSE ↓"), ("psnr", "PSNR (dB) ↑"), ("ssim", "SSIM ↑"), ("time_s", "Time (s) ↓")]
C_KSC, C_PHY, C_RA, C_UNI, C_RND = "#8c8c8c", "#2a6fdb", "#e0702b", "#2ca58d", "#b04fb5"
REF = "PhyKSC-Diff@1000"


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def paired(df, a, b):
    """Aligned per-slice frames for keys a and b."""
    x = df[df.key == a].set_index("idx")
    y = df[df.key == b].set_index("idx")
    common = x.index.intersection(y.index)
    return x.loc[common], y.loc[common]


def two_sided(df, a, b, label, metrics=METRICS):
    rows = []
    x, y = paired(df, a, b)
    for col, name in metrics:
        d = y[col] - x[col]
        res = wilcoxon(x[col], y[col]) if (d != 0).any() else None
        rows.append({"Comparison": label, "Metric": name, "A median": x[col].median(),
                     "B median": y[col].median(), "Median paired diff (B-A)": d.median(),
                     "Wilcoxon W": None if res is None else res.statistic,
                     "p-value": 1.0 if res is None else res.pvalue, "n": len(d)})
    return rows


def noninferiority(df, cand, margins, alpha):
    """Candidate vs REF: one-sided Wilcoxon that the candidate is not worse by more than the margin."""
    ref, c = paired(df, REF, cand)
    d_psnr = c.psnr - ref.psnr                       # >= -m_psnr
    d_ssim = c.ssim - ref.ssim                       # >= -m_ssim
    d_nmse = (c.nmse - ref.nmse) / ref.nmse          # <= m_nmse (relative)
    tests = {"PSNR": d_psnr + margins["psnr"], "SSIM": d_ssim + margins["ssim"],
             "NMSE": margins["nmse"] - d_nmse}
    p = {k: wilcoxon(v, alternative="greater").pvalue for k, v in tests.items()}
    within = (d_psnr >= -margins["psnr"]) & (d_ssim >= -margins["ssim"]) & (d_nmse <= margins["nmse"])
    return {"Method": cand, "n": len(ref), "mean NFE": c.nfe.mean(),
            "NFE reduction (%)": (1 - c.nfe.mean() / ref.nfe.mean()) * 100,
            "Time reduction (%)": (1 - c.time_s.mean() / ref.time_s.mean()) * 100,
            "mean ΔPSNR (dB)": d_psnr.mean(), "mean ΔSSIM": d_ssim.mean(),
            "mean ΔNMSE (%)": d_nmse.mean() * 100,
            "p PSNR": p["PSNR"], "p SSIM": p["SSIM"], "p NMSE": p["NMSE"],
            "slices within tolerance (%)": within.mean() * 100,
            "Non-inferior": "Yes" if max(p.values()) < alpha else "No"}


def fmt(df, floatfmt=".4g"):
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/main")
    ap.add_argument("--train-logs", nargs="*", default=["runs/ksc/train_log.csv", "runs/phyksc/train_log.csv"])
    ap.add_argument("--margin-psnr", type=float, default=0.1, help="dB")
    ap.add_argument("--margin-ssim", type=float, default=0.005)
    ap.add_argument("--margin-nmse", type=float, default=0.02, help="relative increase")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    margins = {"psnr": args.margin_psnr, "ssim": args.margin_ssim, "nmse": args.margin_nmse}
    out_fig = os.path.join(args.res, "figures")
    os.makedirs(out_fig, exist_ok=True)

    raw = pd.read_csv(os.path.join(args.res, "per_slice.csv"))
    raw["key"] = raw.method + "@" + raw["T"].astype(str)
    raw["anatomy"] = np.where(raw.file.str.startswith("file_brain"), "brain", "knee")
    main_keys = ["Zero-filled@0", "KSC-Diff@1000", REF, "RA-PhyKSC-Diff@1000"]
    has = raw[raw.key.isin(main_keys)].groupby("idx").key.nunique()
    df = raw[raw.idx.isin(has[has == len(main_keys)].index)]          # slices with all main configs
    sub_keys = set(raw.key) - set(main_keys)
    has_sub = raw[raw.key.isin(sub_keys)].groupby("idx").key.nunique()
    abl = df[df.idx.isin(has_sub[has_sub == len(sub_keys)].index)]     # ablation subset, complete
    n_main, n_abl = df.idx.nunique(), abl.idx.nunique()
    cols = ["nmse", "psnr", "ssim", "nfe", "time_s"]
    md = [f"# Results — R4, emulated single-coil fastMRI knee + brain\n",
          f"{n_main} test slices ({df[df.key == REF].anatomy.value_counts().to_dict()}); "
          f"ablation / RQ2 subset: {n_abl} slices.  RQ3 margins: PSNR {margins['psnr']} dB, "
          f"SSIM {margins['ssim']}, NMSE +{margins['nmse'] * 100:g}% (relative), α = {args.alpha}.\n"]

    # ============================================================ RQ1
    mean = df.groupby("key")[cols].mean()
    rows = []
    for scope, d in [("all", df)] + list(df.groupby("anatomy")):
        m = d.groupby("key")[cols].mean()
        for key in ["Zero-filled@0", "KSC-Diff@1000", REF]:
            rows.append({"Data": scope, "Method": key.split("@")[0], "n": d.idx.nunique(),
                         "NMSE ↓": m.loc[key, "nmse"], "PSNR (dB) ↑": m.loc[key, "psnr"],
                         "SSIM ↑": m.loc[key, "ssim"], "Time (s)": m.loc[key, "time_s"]})
    t1 = pd.DataFrame(rows)
    md += ["## RQ1 — Table 1. KSC-Diff vs PhyKSC-Diff (DC in training), T=1000\n", fmt(t1, ".5g"), ""]

    rows = []
    for scope, d in [("all", df)] + list(df.groupby("anatomy")):
        for r in two_sided(d, "KSC-Diff@1000", REF, f"KSC vs PhyKSC ({scope})"):
            rows.append(r)
    t2 = pd.DataFrame(rows)
    t2["Holm p"] = holm(t2["p-value"])
    t2["Significant"] = np.where(t2["Holm p"] < args.alpha, "Yes", "No")
    t2 = t2.rename(columns={"A median": "KSC-Diff median", "B median": "PhyKSC-Diff median"})
    md += ["## RQ1 — Table 2. Two-sided Wilcoxon signed-rank, Holm over all rows\n", fmt(t2), ""]

    # ============================================================ RQ3 headline
    ph, ra = mean.loc[REF], mean.loc["RA-PhyKSC-Diff@1000"]
    t3 = pd.DataFrame([ph, ra], index=["PhyKSC-Diff", "RA-PhyKSC-Diff"])[cols]
    t3.loc["Change"] = [ra.nmse - ph.nmse, ra.psnr - ph.psnr, ra.ssim - ph.ssim,
                        (ra.nfe - ph.nfe) / ph.nfe * 100, (ra.time_s - ph.time_s) / ph.time_s * 100]
    t3.columns = ["NMSE ↓", "PSNR (dB) ↑", "SSIM ↑", "NFE ↓ (Change: %)", "Time (s) ↓ (Change: %)"]
    md += ["## RQ3 — Table 3. PhyKSC-Diff vs RA-PhyKSC-Diff, T=1000, all slices\n",
           fmt(t3.reset_index(names="Method"), ".5g"), ""]

    ni = [noninferiority(df, "RA-PhyKSC-Diff@1000", margins, args.alpha)]
    ni[0]["Method"] += " (all slices)"
    if n_abl:
        for key in sorted(sub_keys | {"RA-PhyKSC-Diff@1000"},
                          key=lambda k: (k.split("@")[0], int(k.split("@")[1]))):
            if key != REF:
                ni.append(noninferiority(abl, key, margins, args.alpha))
    t_ni = pd.DataFrame(ni)
    md += ["## RQ3 — Table 4. Non-inferiority vs PhyKSC-Diff@1000 (one-sided Wilcoxon, all three metrics must pass)\n",
           fmt(t_ni), ""]
    ra_pass = t_ni[t_ni.Method.str.startswith("RA-PhyKSC-Diff@") & (t_ni["Non-inferior"] == "Yes")]
    if len(ra_pass):
        best = ra_pass.loc[ra_pass["mean NFE"].idxmin()]
        md += [f"**Smallest-NFE RA configuration that is non-inferior:** {best.Method} — "
               f"{best['mean NFE']:.1f} NFE ({best['NFE reduction (%)']:.1f}% fewer than 1000), "
               f"{best['slices within tolerance (%)']:.1f}% of slices individually within tolerance.\n"]

    if n_abl:
        # ======================================================== Ablation table
        mab = abl.groupby("key")[cols].mean()
        T_list = sorted(int(k.split("@")[1]) for k in mab.index if k.startswith("RA-PhyKSC-Diff@"))
        rows = []
        for T in T_list:
            a, b = mab.loc[f"PhyKSC-Diff@{T}"], mab.loc[f"RA-PhyKSC-Diff@{T}"]
            rows.append({"T": T, "Phy NFE": a.nfe, "Phy NMSE": a.nmse, "Phy PSNR": a.psnr, "Phy SSIM": a.ssim,
                         "Phy Time": a.time_s, "RA NFE": b.nfe, "RA NMSE": b.nmse, "RA PSNR": b.psnr,
                         "RA SSIM": b.ssim, "RA Time": b.time_s})
        t4 = pd.DataFrame(rows)
        md += [f"## Table 5. Diffusion-time ablation, fixed vs residual-adaptive ({n_abl} slices)\n", fmt(t4), ""]

        # ======================================================== RQ2 matched-NFE controls
        ctrl = ["RA-PhyKSC-Diff@1000", "Uniform-matched@1000", "Random-order@1000"]
        t6 = mab.loc[ctrl + [REF]].reset_index(names="Method")
        md += ["## RQ2 — Table 6. Same NFE, different jump placement (subset)\n", fmt(t6, ".5g"), ""]
        rows = two_sided(abl, "Uniform-matched@1000", "RA-PhyKSC-Diff@1000", "Uniform-matched vs RA",
                         METRICS[:3])
        rows += two_sided(abl, "Random-order@1000", "RA-PhyKSC-Diff@1000", "Random-order vs RA", METRICS[:3])
        t7 = pd.DataFrame(rows)
        t7["Holm p"] = holm(t7["p-value"])
        t7["Significant"] = np.where(t7["Holm p"] < args.alpha, "Yes", "No")
        t7 = t7.rename(columns={"A median": "Control median", "B median": "RA median"})
        md += ["## RQ2 — Table 7. RA vs matched-NFE controls (two-sided Wilcoxon, Holm)\n", fmt(t7), ""]

        thr = [k for k in mab.index if "(thr x" in k]
        scale = lambda k: float(k.split("thr x")[1].split(")")[0])
        t8 = mab.loc[["RA-PhyKSC-Diff@1000"] + sorted(thr, key=scale)].reset_index(names="Controller")
        t8.insert(1, "thresholds (high, low)",
                  ["(0.02, 0.005)"] + [f"({0.02 * scale(k):g}, {0.005 * scale(k):g})" for k in sorted(thr, key=scale)])
        md += ["## RQ2 — Table 8. Controller threshold sensitivity (subset, T=1000)\n", fmt(t8, ".5g"), ""]

    # ============================================================ RQ2 probe
    probe_path = os.path.join(args.res, "probe.csv")
    pr = pd.read_csv(probe_path).dropna(subset=["dR"]) if os.path.exists(probe_path) else pd.DataFrame()
    if len(pr):
        rows = []
        for j in (5, 10):
            d = pr.dropna(subset=[f"dev{j}"])
            rho = spearmanr(d.dR, d[f"dev{j}"])
            per = d.groupby("idx")[["dR", f"dev{j}"]].apply(lambda g: spearmanr(g.iloc[:, 0], g.iloc[:, 1]).statistic)
            costly = d[f"dev{j}"] > d[f"dev{j}"].median()
            rows.append({"jump": f"Δt={j}", "states": len(d), "slices": d.idx.nunique(),
                         "Spearman ρ (pooled)": rho.statistic, "p": rho.pvalue,
                         "median per-slice ρ": per.median(),
                         "AUC: ΔR predicts above-median skip error": roc_auc_score(costly, d.dR)})
        t9 = pd.DataFrame(rows)
        md += ["## RQ2 — Table 9. Does ΔR_t predict the error of skipping? (probe on the fixed trajectory)\n",
               fmt(t9), ""]
        cls = pd.cut(pr.dR, [-np.inf, 0.005, 0.02, np.inf], labels=["ΔR ≤ 0.005 (→ Δt=10)",
                                                                    "0.005 < ΔR ≤ 0.02 (→ Δt=5)",
                                                                    "ΔR > 0.02 (→ Δt=1)"])
        t10 = pr.groupby(cls, observed=False).agg(states=("dR", "size"), mean_dev5=("dev5", "mean"),
                                                  median_dev10=("dev10", "median"),
                                                  mean_dev10=("dev10", "mean")).reset_index(names="controller region")
        md += ["## RQ2 — Table 10. Skip error by the region the controller would assign\n", fmt(t10), ""]

    # RA step usage
    traj = {}
    for fpath in glob.glob(os.path.join(args.res, "trajectories", "*.json")):
        T = int(fpath.rsplit("_T", 1)[1].split(".")[0])
        traj.setdefault(T, []).append(json.load(open(fpath)))
    rows = []
    for T in sorted(traj):
        jumps = np.concatenate([[round((h[0] - h[1]) / (1000 / T)) for h in tr] for tr in traj[T]])
        rows.append({"T": T, "slices": len(traj[T]), "Δt=1 (%)": np.mean(jumps == 1) * 100,
                     "Δt=5 (%)": np.mean(jumps == 5) * 100, "Δt=10 (%)": np.mean(jumps == 10) * 100,
                     "mean NFE": np.mean([len(t) for t in traj[T]])})
    md += ["## RA controller step usage\n", fmt(pd.DataFrame(rows)), ""]

    with open(os.path.join(args.res, "RESULTS.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    for name, t in [("table1_rq1_means", t1), ("table2_rq1_stats", t2), ("table3_rq3", t3),
                    ("table4_rq3_noninferiority", t_ni)] + \
            ([("table5_ablation", t4), ("table6_rq2_controls", t6), ("table7_rq2_control_stats", t7),
              ("table8_rq2_thresholds", t8)] if n_abl else []) + \
            ([("table9_rq2_probe", t9), ("table10_rq2_probe_regions", t10)] if len(pr) else []):
        t.to_csv(os.path.join(args.res, f"{name}.csv"), index=False)

    # ============================================================ figures
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    # RQ1 bars per anatomy
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    groups = ["all", "knee", "brain"]
    for ax, (col, name) in zip(axs, [("nmse", "NMSE ↓"), ("psnr", "PSNR (dB) ↑"), ("ssim", "SSIM ↑")]):
        for i, (meth, c) in enumerate([("KSC-Diff", C_KSC), ("PhyKSC-Diff", C_PHY)]):
            vals = [t1[(t1.Data == g) & (t1.Method == meth)][name].iloc[0] for g in groups]
            bars = ax.bar(np.arange(3) + (i - 0.5) * 0.38, vals, 0.38, color=c, label=meth)
            ax.bar_label(bars, fmt="%.3g", fontsize=7)
        ax.set_xticks(range(3), groups); ax.set_title(name)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_fig, "fig4_rq1_dc_training.png"), dpi=200); plt.close(fig)

    # qualitative
    recs = sorted(glob.glob(os.path.join(args.res, "recon", "*.npz")))
    keys = ["gt", "Zero-filled@0", "KSC-Diff@1000", REF, "RA-PhyKSC-Diff@1000"]
    recs = [r for r in recs if set(keys) <= set(np.load(r).files)]
    if recs:
        titles = ["Ground truth", "Zero-filled R4", "KSC-Diff", "PhyKSC-Diff", "RA-PhyKSC-Diff"]
        fig, axs = plt.subplots(2 * len(recs), 5, figsize=(12, 4.9 * len(recs)), squeeze=False)
        for r, path in enumerate(recs):
            d = np.load(path)
            vmax = d["gt"].max()
            for c, (key, title) in enumerate(zip(keys, titles)):
                axs[2 * r, c].imshow(d[key], cmap="gray", vmin=0, vmax=vmax)
                axs[2 * r, c].set_title(title if r == 0 else "")
                axs[2 * r + 1, c].imshow(np.abs(d[key] - d["gt"]), cmap="inferno", vmin=0, vmax=0.2 * vmax)
                for a in (axs[2 * r, c], axs[2 * r + 1, c]):
                    a.set_xticks([]); a.set_yticks([])
            axs[2 * r, 0].set_ylabel(os.path.basename(path)[:-4]); axs[2 * r + 1, 0].set_ylabel("|error|")
        fig.tight_layout(); fig.savefig(os.path.join(out_fig, "fig5_qualitative.png"), dpi=130); plt.close(fig)

    if n_abl:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot(t4["T"], t4["Phy NFE"], "o-", color=C_PHY, label="PhyKSC-Diff")
        ax.plot(t4["T"], t4["RA NFE"], "s-", color=C_RA, label="RA-PhyKSC-Diff")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Diffusion timesteps T"); ax.set_ylabel("Mean NFE")
        ax.legend(frameon=False); fig.tight_layout()
        fig.savefig(os.path.join(out_fig, "fig6_nfe_vs_T.png"), dpi=200); plt.close(fig)

        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, (col, name) in zip(axs, [("psnr", "Mean PSNR (dB)"), ("ssim", "Mean SSIM")]):
            ax.plot(t4["Phy NFE"], t4[f"Phy {col.upper()}"], "o-", color=C_PHY, label="PhyKSC-Diff (fixed T)")
            ax.plot(t4["RA NFE"], t4[f"RA {col.upper()}"], "s-", color=C_RA, label="RA-PhyKSC-Diff")
            for key, c, mk, lab in [("Uniform-matched@1000", C_UNI, "^", "Uniform, NFE-matched"),
                                    ("Random-order@1000", C_RND, "D", "Random-order, NFE-matched")]:
                ax.plot(mab.loc[key, "nfe"], mab.loc[key, col], mk, color=c, ms=8, label=lab)
            for k in thr:
                ax.plot(mab.loc[k, "nfe"], mab.loc[k, col], "x", color=C_RA, ms=7)
                ax.annotate(k.split("(")[1].rstrip(")@1000").rstrip(")"), (mab.loc[k, "nfe"], mab.loc[k, col]),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_xscale("log"); ax.set_xlabel("Mean NFE"); ax.set_ylabel(name)
        axs[0].legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(out_fig, "fig7_quality_vs_nfe.png"), dpi=200); plt.close(fig)

    if 1000 in traj:
        tr = traj[1000][0]
        ts = [1000] + [h[1] for h in tr]
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.plot(range(len(ts)), ts, color=C_RA, label=f"RA-PhyKSC-Diff ({len(tr)} NFE)")
        ax.plot(range(1001), np.arange(1000, -1, -1), "--", color=C_PHY, label="PhyKSC-Diff (Δt = 1)")
        ax.set_xlabel("Reverse iteration (NFE)"); ax.set_ylabel("Timestep t")
        ax.legend(frameon=False); fig.tight_layout()
        fig.savefig(os.path.join(out_fig, "fig8_trajectory.png"), dpi=200); plt.close(fig)

    if len(pr):
        d = pr.dropna(subset=["dev10"])
        d = d[(d.dR > 0) & (d.dev10 > 0)]
        bins = np.logspace(np.log10(d.dR.quantile(0.01)), np.log10(d.dR.quantile(0.99)), 25)
        b = pd.cut(d.dR, bins)
        g = d.groupby(b, observed=True).dev10.agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        mid = [iv.mid for iv in g.index]
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.scatter(d.dR, d.dev10, s=2, alpha=0.08, color="#999")
        ax.plot(mid, g["median"], "o-", color=C_RA, label="median (binned)")
        ax.fill_between(mid, g.iloc[:, 1], g.iloc[:, 2], color=C_RA, alpha=0.2, label="IQR")
        for x in (0.005, 0.02):
            ax.axvline(x, color="k", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("ΔR_t (relative residual change)"); ax.set_ylabel("deviation of a Δt=10 jump")
        ax.legend(frameon=False, fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(out_fig, "fig9_rq2_probe.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for path, c, lab in zip(args.train_logs, [C_KSC, C_PHY], ["KSC-Diff", "PhyKSC-Diff (DC loss)"]):
        if os.path.exists(path):
            lg = pd.read_csv(path)
            ax.plot(lg.epoch, lg.loss, color=c, label=lab)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train L1 loss"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out_fig, "train_loss.png"), dpi=200); plt.close(fig)

    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
