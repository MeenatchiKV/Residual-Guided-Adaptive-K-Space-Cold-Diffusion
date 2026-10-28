"""PhyKSC-Diff architecture diagram with data consistency in training.

    python figures/make_phyksc_diagram.py   ->  figures/phyksc_dc_training.{png,svg,pdf}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "stix", "svg.fonttype": "none"})

W, H = 1400, 760
BLACK, BLUE, BLUE_BG = "#111111", "#1f5fbf", "#eef3fc"
fig = plt.figure(figsize=(W / 100, H / 100))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def box(x, y, w, h, r=18, color=BLACK, fill="white", lw=1.4, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                ec=color, fc=fill, lw=lw, ls=ls))


def text(x, y, s, size=11, weight="normal", color=BLACK, ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, fontstyle=style)


def arrow(p, q, color=BLACK, ls="-", lw=1.4, conn="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=conn, shrinkA=0, shrinkB=0))


def path_arrow(points, color=BLACK, ls="-", lw=1.4):
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, solid_capstyle="butt")
    arrow(points[-2], points[-1], color=color, ls=ls, lw=lw)


# ---------------------------------------------------------------- title
box(10, 10, 1380, 72, r=14)
text(700, 33, "PhyKSC Diffusion: Physics K-Space Cold Diffusion Architecture", 15, "bold")
text(700, 60, "Learned K-Space Cold-Diffusion Prior + Explicit MRI Physics via Data Consistency in Training",
     15, "bold")

# ---------------------------------------------------------------- left column
box(20, 102, 390, 168, r=22)
text(215, 132, "Undersampling", 17, "bold")
text(215, 172, r"Single-coil k-space $K_0$ and sampling mask $M$", 11.5)
text(215, 212, r"$y = M \odot K_0$", 15)
text(215, 246, "(acquired columns, kept fixed at every $t$)", 10, style="italic")
arrow((215, 270), (215, 300))

box(20, 300, 390, 390, r=30)
text(215, 335, "Cold Diffusion", 17, "bold")
text(215, 362, "Forward Process", 17, "bold")
text(215, 390, "(Training only)", 12, "bold")
text(215, 428, r"Fully sampled k-space $K_0$", 11.5)
box(55, 455, 320, 62, r=8)
text(215, 476, "Deterministic k-space degradation", 11.5)
text(215, 499, r"$D_t(\cdot)$: removes unacquired columns only", 11)
text(215, 552, r"$K_0 \rightarrow K_1 \rightarrow \cdots \rightarrow K_T = y$", 14)
text(215, 600, r"$x_t = \mathcal{F}^{-1} D_t(K_0)$,   $t \sim \mathcal{U}\{0,\ldots,T\}$", 13)
text(215, 645, r"target  $x_0 = \mathcal{F}^{-1} K_0$", 13)

# ---------------------------------------------------------------- denoising network
box(440, 102, 410, 588, r=34)
text(645, 135, "Denoising Network", 17, "bold")
text(645, 168, "Training objective (DC inside the loss):", 11.5)
text(645, 205, r"$\mathbb{E}_{K_0,t}\,\left\|\,\mathcal{F}^{-1}\,\mathrm{DC}\left(\mathcal{F}\, f_\theta(x_t,t)\right) - x_0\right\|_1$",
     14.5, color=BLUE)
text(512, 262, "Timestep", 12)
text(512, 281, r"$t$", 13)
box(590, 252, 200, 52, r=8)
text(690, 278, "Time Embedding", 11)
arrow((545, 270), (590, 270))
box(640, 336, 100, 50, r=8)
text(690, 361, "MLP", 12)
arrow((690, 304), (690, 336))
box(595, 425, 190, 90, r=12)
text(690, 458, "2D U-Net with", 12)
text(690, 480, "Skip Connections", 12)
arrow((690, 386), (690, 425))
text(512, 462, r"Input $x_t$", 11)
text(512, 482, r"$(1 \times H \times W \times 2)$", 10.5)
arrow((410, 470), (595, 470))
arrow((690, 515), (690, 565))
text(690, 590, r"Predicted clean image $\hat{x}_0 = f_\theta(x_t,t)$", 11.5)
text(690, 612, r"$(1 \times H \times W \times 2)$", 10.5)
text(645, 660, "gradients reach the U-Net only through", 10, color=BLUE, style="italic")
text(645, 677, "the unacquired k-space columns", 10, color=BLUE, style="italic")

# network output -> DC layer (leaves the U-Net on its right, above the gradient path)
path_arrow([(785, 445), (865, 445), (865, 215), (880, 215)])
text(822, 432, r"$\hat{x}_0$", 12)

# ---------------------------------------------------------------- DC layer (training + inference)
box(880, 102, 500, 232, r=22, color=BLUE, fill=BLUE_BG, lw=2.2)
text(1130, 133, "Physics-based Data Consistency Layer", 17, "bold")
text(1130, 160, "(Training and inference)", 12, "bold", color=BLUE)
box(900, 190, 120, 56, r=8)
text(960, 210, "FFT", 11.5)
text(960, 230, r"$\hat{k} = \mathcal{F}\hat{x}_0$", 12)
box(1048, 182, 200, 72, r=8, color=BLUE, lw=1.8)
text(1148, 203, "Apply Data Consistency:", 11, "bold")
text(1148, 233, r"$\hat{k}^{DC} = M \odot y + (1-M) \odot \hat{k}$", 12.5)
box(1276, 190, 90, 56, r=8)
text(1321, 210, "IFFT", 11.5)
text(1321, 230, r"$\hat{x}_0^{DC}$", 12.5)
arrow((1020, 218), (1048, 218))
arrow((1248, 218), (1276, 218))
text(1130, 284, "acquired columns are replaced by the measurement $y$;", 10.5, style="italic")
text(1130, 304, "the network output is kept only where $M = 0$", 10.5, style="italic")

# ---------------------------------------------------------------- training loss (training only)
box(880, 372, 240, 318, r=22, color=BLUE, fill=BLUE_BG, lw=2.2)
text(1000, 405, "Training Loss", 17, "bold")
text(1000, 431, "(Training only)", 12, "bold", color=BLUE)
text(1000, 490, r"$\mathcal{L}(\theta) = \left\| \hat{x}_0^{DC} - x_0 \right\|_1$", 15)
text(1000, 545, "computed on the data-", 11)
text(1000, 564, "consistent prediction", 11)
text(1000, 615, r"backpropagate $\nabla_\theta \mathcal{L}$", 11.5, color=BLUE)
text(1000, 636, "through DC layer to U-Net", 11.5, color=BLUE)
path_arrow([(1000, 334), (1000, 372)], color=BLUE, lw=2.0)
text(965, 352, "training", 10, color=BLUE, style="italic")
# gradient back to the U-Net (dashed)
path_arrow([(880, 495), (785, 495)], color=BLUE, ls="--", lw=1.6)
text(826, 512, r"$\nabla_\theta$", 12, color=BLUE)

# ---------------------------------------------------------------- inference
box(1140, 372, 240, 188, r=22)
text(1260, 400, "Reverse Cold Diffusion", 15, "bold")
text(1260, 424, "(Inference only)", 12, "bold")
text(1260, 461, r"$x_{t-1} = x_t - \mathcal{F}^{-1} D_t(\hat{k}^{DC})$", 12.5)
text(1268, 489, r"$+\, \mathcal{F}^{-1} D_{t-1}(\hat{k}^{DC})$", 12.5)
text(1260, 526, r"Repeat for $t = T, \ldots, 1$", 11.5)
text(1260, 545, r"(next input $x_{t-1}$)", 10.5, style="italic")
path_arrow([(1260, 334), (1260, 372)], lw=1.6)
text(1300, 352, "inference", 10, style="italic")

box(1140, 580, 240, 110, r=22)
text(1260, 606, "Reconstruction Output", 14, "bold")
box(1224, 626, 72, 36, r=6)
text(1260, 644, "IFFT", 11)
text(1176, 644, "k-space", 10.5)
text(1344, 644, "image", 10.5)
arrow((1204, 644), (1224, 644)); arrow((1296, 644), (1320, 644))
path_arrow([(1260, 560), (1260, 580)], lw=1.6)

# ---------------------------------------------------------------- legend
ax.plot([30, 80], [728, 728], color=BLUE, lw=2.2)
text(90, 728, "data consistency in training (new in PhyKSC-Diff)", 11, ha="left")
ax.plot([470, 520], [728, 728], color=BLUE, lw=1.6, ls="--")
text(530, 728, "gradient path", 11, ha="left")
ax.plot([680, 730], [728, 728], color=BLACK, lw=1.4)
text(740, 728, "data flow", 11, ha="left")

out = os.path.dirname(os.path.abspath(__file__))
for ext in ("png", "svg", "pdf"):
    fig.savefig(os.path.join(out, f"phyksc_dc_training.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(out, "phyksc_dc_training.*"))
