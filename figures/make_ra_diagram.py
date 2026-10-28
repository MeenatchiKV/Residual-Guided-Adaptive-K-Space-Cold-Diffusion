"""Figure 4: one RA-PhyKSC-Diff reverse iteration, with real intermediate states.

    python figures/capture_ra_iteration.py      (once, writes figures/ra_iteration.npz)
    python figures/make_ra_diagram.py           ->  figures/raphyksc_iteration.{png,svg,pdf}
Equations follow Section 3.3 (iteration index i, f_theta, y, F^-1 on the D terms).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "stix", "svg.fonttype": "none"})
here = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(here, "ra_iteration.npz"))

W, H = 1500, 760
BLACK, GREY, BLUE, BLUE_BG = "#111111", "#666666", "#1f5fbf", "#eef3fc"
fig = plt.figure(figsize=(W / 100, H / 100))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def box(x, y, w, h, r=14, color=BLACK, fill="white", lw=1.4, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                ec=color, fc=fill, lw=lw, ls=ls))


def text(x, y, s, size=11, weight="normal", color=BLACK, ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, fontstyle=style)


def arrow(p, q, color=BLACK, lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


def path_arrow(points, color=BLACK, lw=1.4):
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw)
    arrow(points[-2], points[-1], color=color, lw=lw)


def thumb(img, x, y, s=92, kind="image"):
    """Place an image thumbnail with its top-left corner at data coords (x, y)."""
    a = fig.add_axes([x / W, 1 - (y + s) / H, s / W, s / H])
    if kind == "kspace":
        img = np.log1p(img / np.percentile(img, 99) * 50)
        a.imshow(img, cmap="gray", vmin=0, vmax=img.max())
    elif kind == "mask":
        a.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    else:
        a.imshow(img, cmap="gray", vmin=0, vmax=np.percentile(d["x_dc"], 99.7))
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_linewidth(0.8)


# ---------------------------------------------------------------- title
box(380, 8, 740, 44, r=10)
text(750, 30, "RA-PhyKSC-Diff: one reverse cold-diffusion iteration $i$", 15, "bold")

# ---------------------------------------------------------------- top row: steps 1-5
TY = 120
text(70, TY - 22, r"$x^{(i)}$", 15)
text(70, TY + 112, "state at level $t_i$", 9.5)
thumb(d["x_old"], 24, TY - 6)

box(140, TY - 28, 190, 160, color=BLUE, fill=BLUE_BG, lw=1.8)
text(235, TY - 8, "1. Network prediction", 11.5, "bold")
text(235, TY + 12, "(PhyKSC-Diff, frozen)", 9.5, color=BLUE)
text(235, TY + 55, r"$\hat{x}_0^{(i)} = f_\theta\!\left(x^{(i)}, t_i\right)$", 14)
text(235, TY + 100, "no retraining", 9.5, style="italic", color=GREY)
arrow((116, TY + 40), (140, TY + 40))
thumb(d["x_hat"], 350, TY - 6)
text(396, TY + 112, r"$\hat{x}_0^{(i)}$", 12)
arrow((330, TY + 40), (350, TY + 40))

box(462, TY - 28, 162, 160)
text(543, TY - 8, "2. Fourier transform", 11.5, "bold")
text(543, TY + 22, r"$\hat{k}_i = \mathcal{F}\hat{x}_0^{(i)}$", 13.5)
thumb(d["k_hat"], 512, TY + 36, s=62, kind="kspace")
arrow((442, TY + 40), (462, TY + 40))

box(644, TY - 28, 250, 160, color=BLUE, fill=BLUE_BG, lw=1.8)
text(769, TY - 8, "3. Data consistency (first DC)", 11.5, "bold")
text(769, TY + 22, r"$\hat{k}_i^{\mathrm{DC}} = M \odot y + (1-M) \odot \hat{k}_i$", 12.5)
thumb(d["y"], 680, TY + 40, s=62, kind="kspace")
text(711, TY + 116, r"$y$ (acquired)", 9.5)
thumb(d["mask"], 796, TY + 40, s=62, kind="mask")
text(827, TY + 116, r"$M$ (R4 mask)", 9.5)
arrow((624, TY + 40), (644, TY + 40))

box(912, TY - 28, 128, 160)
text(976, TY - 8, "4. Inverse FT", 11.5, "bold")
text(976, TY + 40, r"$x_i^{\mathrm{DC}} = \mathcal{F}^{-1}\hat{k}_i^{\mathrm{DC}}$", 12.5)
arrow((894, TY + 40), (912, TY + 40))
thumb(d["x_dc"], 1056, TY - 6)
text(1102, TY + 112, r"$x_i^{\mathrm{DC}}$", 12)
arrow((1040, TY + 40), (1056, TY + 40))

# step 5 (right, spans both rows)
box(1162, 70, 328, 430, color=BLUE, fill=BLUE_BG, lw=1.8)
text(1326, 92, "5. Residual and step-size selection", 11.5, "bold")
box(1178, 110, 296, 78, r=8)
text(1326, 126, "reconstruction-change residual", 9.5, color=GREY)
text(1326, 160, r"$R_i = \dfrac{\| x_i^{\mathrm{DC}} - x^{(i)} \|_2}{\| x^{(i)} \|_2 + \varepsilon}$", 13)
box(1178, 198, 296, 70, r=8)
text(1326, 213, "relative change", 9.5, color=GREY)
text(1326, 243, r"$\Delta R_i = \dfrac{| R_i - R_{i-1} |}{R_{i-1} + \varepsilon}$", 13)
box(1178, 278, 296, 110, r=8)
text(1326, 293, "step size (levels)", 9.5, color=GREY)
text(1326, 312, r"$\Delta t_i = 1$   if $\Delta R_i > 0.02$", 12)
text(1326, 338, r"$\Delta t_i = 5$   if $0.005 < \Delta R_i \leq 0.02$", 12)
text(1326, 364, r"$\Delta t_i = 10$  if $\Delta R_i \leq 0.005$", 12)
text(1326, 404, r"$t_{i+1} = \max(0,\, t_i - \Delta t_i)$", 13)
text(1326, 437, "example shown (knee test slice):", 9.5, style="italic", color=GREY)
text(1326, 457, rf"$t_i = {int(d['t'])}$,  $R_i = {float(d['R']):.3f}$,  $\Delta R_i = {float(d['dR']):.4f}$", 10.5)
text(1326, 479, rf"$\Rightarrow\ \Delta t_i = {int(d['jump'])}$,  $t_{{i+1}} = {int(d['t_next'])}$", 10.5)
arrow((1148, TY + 40), (1162, TY + 40))
ax.plot([1142, 1142], [TY + 40, TY + 40], color=BLACK)

# ---------------------------------------------------------------- bottom row: steps 6-8 (right to left)
BY = 560
box(860, BY - 30, 285, 162)
text(1002, BY - 10, "6. Cold-diffusion step (jump)", 11.5, "bold")
text(1002, BY + 30, r"$\tilde{x}^{(i+1)} = x^{(i)} - \mathcal{F}^{-1} D_{t_i}\!\left(\hat{k}_i^{\mathrm{DC}}\right)$", 12.5)
text(1028, BY + 62, r"$+\ \mathcal{F}^{-1} D_{t_{i+1}}\!\left(\hat{k}_i^{\mathrm{DC}}\right)$", 12.5)
text(1002, BY + 98, "re-introduces the columns removed between", 9.5, style="italic", color=GREY)
text(1002, BY + 113, r"levels $t_i$ and $t_{i+1}$; no extra network evaluation", 9.5, style="italic", color=GREY)
path_arrow([(1326, 500), (1326, BY + 45), (1145, BY + 45)])

box(520, BY - 30, 300, 150, color=BLUE, fill=BLUE_BG, lw=1.8)
text(670, BY - 10, "7. Data consistency (second DC)", 11.5, "bold")
text(670, BY + 40, r"$k^{(i+1)} = M \odot y + (1-M) \odot \mathcal{F}\tilde{x}^{(i+1)}$", 12.5)
text(670, BY + 88, "acquired columns are unchanged by step 6;", 9.5, style="italic", color=GREY)
text(670, BY + 103, "kept as a numerical safeguard", 9.5, style="italic", color=GREY)
arrow((860, BY + 45), (820, BY + 45))

box(300, BY - 30, 180, 150)
text(390, BY - 10, "8. Inverse FT", 11.5, "bold")
text(390, BY + 40, r"$x^{(i+1)} = \mathcal{F}^{-1} k^{(i+1)}$", 13)
arrow((520, BY + 45), (480, BY + 45))

thumb(d["x_next"], 150, BY - 1)
text(196, BY - 18, r"$x^{(i+1)}$", 15)
text(196, BY + 108, rf"next state, level $t_{{i+1}}$", 9.5)
arrow((300, BY + 45), (242, BY + 45))

# loop back to the next iteration
path_arrow([(150, BY + 45), (70, BY + 45), (70, TY + 128)], color=GREY)
text(84, BY - 60, r"$i \leftarrow i+1$", 12, ha="left", color=GREY)
text(84, BY - 38, "until $t_i = 0$", 10.5, ha="left", color=GREY)
text(750, 735, "Blue: PhyKSC-Diff network with data consistency (as trained).  NFE = number of iterations; "
     "the residual and the jump reuse tensors computed in steps 1-4.", 10, color=GREY)

for ext in ("png", "svg", "pdf"):
    fig.savefig(os.path.join(here, f"raphyksc_iteration.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(here, "raphyksc_iteration.*"))
