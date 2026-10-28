"""Figure 2: KSC-Diff architecture (pipeline + the U-Net actually used, ksc/unet.py).

    python figures/make_kscdiff_diagram.py   ->  figures/kscdiff_architecture.{png,svg,pdf}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "stix", "svg.fonttype": "none"})

W, H = 1400, 1010
BLACK, GREY, ENC, DEC, MID = "#111111", "#666666", "#dfe7f3", "#f3e6d8", "#e4e4e4"
fig = plt.figure(figsize=(W / 100, H / 100))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def box(x, y, w, h, r=18, color=BLACK, fill="white", lw=1.4, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                ec=color, fc=fill, lw=lw, ls=ls))


def text(x, y, s, size=11, weight="normal", color=BLACK, ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, fontstyle=style)


def arrow(p, q, color=BLACK, ls="-", lw=1.4, conn="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=conn, shrinkA=0, shrinkB=0))


def path_arrow(points, color=BLACK, ls="-", lw=1.4):
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls)
    arrow(points[-2], points[-1], color=color, ls=ls, lw=lw)


def block(cx, cy, w, h, fill, depth=10):
    """Pseudo-3D feature block centred at (cx, cy)."""
    x, y = cx - w / 2, cy - h / 2
    ax.add_patch(Polygon([(x, y), (x + depth, y - depth), (x + w + depth, y - depth), (x + w, y)],
                         closed=True, fc="#ffffff", ec=BLACK, lw=1))
    ax.add_patch(Polygon([(x + w, y), (x + w + depth, y - depth), (x + w + depth, y + h - depth), (x + w, y + h)],
                         closed=True, fc="#c9c9c9", ec=BLACK, lw=1))
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], closed=True, fc=fill, ec=BLACK, lw=1))


# ---------------------------------------------------------------- title
box(10, 10, 1380, 52, r=12)
text(700, 36, "KSC Diffusion: K-Space Cold Diffusion Architecture", 16, "bold")

# ---------------------------------------------------------------- pipeline (top)
box(20, 80, 300, 160, r=20)
text(170, 106, "Undersampling", 15, "bold")
text(170, 140, r"Single-coil k-space $K_0$, mask $M$", 11)
text(170, 172, r"$y = M \odot K_0$", 14)
text(170, 205, "R4 random column mask,", 10, style="italic")
text(170, 222, "centre fraction 0.08", 10, style="italic")
arrow((170, 240), (170, 262))

box(20, 262, 300, 300, r=24)
text(170, 290, "Cold Diffusion", 15, "bold")
text(170, 312, "Forward Process", 15, "bold")
text(170, 335, "(Training only)", 11.5, "bold")
box(38, 352, 264, 58, r=7)
text(170, 371, r"Deterministic degradation $D_t(\cdot)$:", 11)
text(170, 392, "removes unacquired columns only", 10.5)
text(170, 437, r"$K_0 \rightarrow K_1 \rightarrow \cdots \rightarrow K_T = y$", 13)
text(170, 480, r"$x_t = \mathcal{F}^{-1} D_t(K_0)$", 13)
text(170, 510, r"$t \sim \mathcal{U}\{0,\ldots,T\}$,  target $x_0 = \mathcal{F}^{-1}K_0$", 11.5)
text(170, 540, "fixed random removal order per sample", 9.5, style="italic")

box(345, 80, 370, 482, r=28)
text(530, 108, "Denoising Network", 15, "bold")
text(530, 136, r"Training objective:", 11)
text(530, 166, r"$\mathbb{E}_{K_0,t}\,\left\|\,f_\theta(x_t,t) - x_0\right\|_1$", 14)
text(415, 220, "Timestep", 11); text(415, 238, r"$t$", 12)
box(470, 208, 150, 44, r=7); text(545, 230, "Sinusoidal embedding", 10)
arrow((440, 230), (470, 230))
box(495, 282, 100, 40, r=7); text(545, 302, "MLP", 11)
arrow((545, 252), (545, 282))
box(460, 360, 170, 70, r=10); text(545, 385, "Time-conditioned", 11); text(545, 405, "2D U-Net (below)", 11)
arrow((545, 322), (545, 360))
text(400, 360, r"Input $x_t$", 10.5); text(400, 379, r"$(2 \times H \times W)$", 10)
arrow((320, 395), (460, 395))
arrow((545, 430), (545, 470))
text(545, 492, r"Predicted clean image $\hat{x}_0 = f_\theta(x_t,t)$", 11)
text(545, 512, r"$(2 \times H \times W)$, real and imaginary parts", 10)
text(530, 545, "no noise is added or predicted", 10, style="italic", color=GREY)

box(740, 80, 400, 300, r=24)
text(940, 108, "Reverse Cold Diffusion", 15, "bold")
text(940, 131, "(Inference only)", 11.5, "bold")
text(940, 166, r"start: $x_T = \mathcal{F}^{-1} y$  (zero-filled)", 12)
text(940, 205, r"$\hat{k}_t = \mathcal{F}\, f_\theta(x_t, t)$", 13)
text(940, 245, r"$x_{t-1} = x_t - \mathcal{F}^{-1} D_t(\hat{k}_t)$", 13)
text(950, 275, r"$+\, \mathcal{F}^{-1} D_{t-1}(\hat{k}_t)$", 13)
text(940, 312, r"repeat for $t = T, \ldots, 1$", 11.5)
text(940, 340, "acquired columns stay equal to $y$", 10, style="italic", color=GREY)
path_arrow([(715, 300), (740, 300)])

box(1160, 80, 220, 300, r=24)
text(1270, 112, "Reconstruction", 14, "bold")
text(1270, 134, "Output", 14, "bold")
text(1270, 185, r"$x_0$ after the", 11.5); text(1270, 205, "last reverse step", 11.5)
box(1215, 235, 110, 50, r=8); text(1270, 260, r"$|\cdot|$  magnitude", 11)
arrow((1270, 210 + 3), (1270, 235))
text(1270, 320, "Reconstructed", 11); text(1270, 340, "image", 11)
arrow((1270, 285), (1270, 305))
arrow((1140, 230), (1160, 230))

box(740, 400, 640, 162, r=20, color=GREY, ls="--", lw=1.2)
text(1060, 425, "Diffusion-time configuration", 12.5, "bold")
text(1060, 458, r"$T_{\max} = 1000$ degradation levels; a configuration $T$ traverses", 11)
text(1060, 480, r"$T$ evenly spaced levels from $t = T_{\max}$ to $0$ with the same network", 11)
text(1060, 512, r"fixed-step sampler: NFE $= T$", 11.5)
text(1060, 538, "same network for every configuration", 10, style="italic", color=GREY)

# ---------------------------------------------------------------- U-Net (bottom)
box(20, 585, 1360, 410, r=26)
text(700, 610, "Time-conditioned 2D U-Net  (56.6 M parameters)", 14, "bold")
base = 790
enc = [(140, 64, 320, 150), (300, 128, 160, 112), (450, 256, 80, 80), (590, 512, 40, 56)]
dec = [(990, 256, 40, 56), (1110, 128, 80, 80), (1230, 64, 160, 112)]   # resolution where the skip is joined
for cx, ch, res, h in enc:
    block(cx, base, 46 + ch / 16, h, ENC)
    text(cx, base + h / 2 + 18, f"{ch}", 11, "bold"); text(cx, base + h / 2 + 34, f"{res}²", 9.5)
block(780, base, 82, 56, MID)
text(780, base + 46, "512", 11, "bold"); text(780, base + 62, "40²", 9.5)
for cx, ch, res, h in dec:
    block(cx, base, 46 + ch / 16, h, DEC)
    text(cx, base + h / 2 + 18, f"{ch}", 11, "bold"); text(cx, base + h / 2 + 34, f"{res}²", 9.5)
# data flow
text(60, base - 8, r"$x_t$", 12); text(60, base + 12, r"$2\times320^2$", 9)
arrow((80, base), (113, base))
for (a, _, _, _), (b, _, _, _) in zip(enc[:-1], enc[1:]):
    arrow((a + 32, base), (b - 30, base))
    text((a + b) / 2, base - 12, "↓ 2", 9, color=GREY)
arrow((620, base), (734, base))
arrow((826, base), (960, base))
arrow((1022, base), (1082, base)); text(1052, base - 12, "↑ 2", 9, color=GREY)
arrow((1142, base), (1200, base)); text(1171, base - 12, "↑ 2", 9, color=GREY)
arrow((1262, base), (1330, base)); text(1296, base - 12, "↑ 2", 9, color=GREY)
text(1352, base - 8, r"$\hat{x}_0$", 12); text(1352, base + 12, r"$2\times320^2$", 9)
# skip connections (the 64-channel encoder output is not reused, as in the implementation)
for (ex, _, _, eh), (dx, _, _, dh) in zip(enc[1:][::-1], dec):
    top = base - max(eh, dh) / 2 - 22 - (dh - 56) * 0.55
    ax.plot([ex, ex, dx], [base - eh / 2 - 10, top, top], color=BLACK, lw=1, ls=":")
    arrow((dx, top), (dx, base - dh / 2 - 10), lw=1, ls=":")
text(820, 658, "skip connections (concatenation)", 10, style="italic")
text(345, base + 138, "Encoder: 2 × ConvNeXt block + linear attention per level", 10.5)
text(780, base + 138, "Bottleneck: ConvNeXt +", 10.5); text(780, base + 154, "attention + ConvNeXt", 10.5)
text(1130, base + 138, "Decoder: skip concatenated, 2 × ConvNeXt block +", 10.5)
text(1130, base + 154, "linear attention, then ↑ 2; final ConvNeXt + 1×1 conv", 10.5)
text(700, 968, "ConvNeXt block: depthwise 7×7 conv + time embedding, LayerNorm, 3×3 conv, GELU, 3×3 conv, "
     "residual.   ↓ 2: 4×4 stride-2 conv;  ↑ 2: 4×4 stride-2 transposed conv.", 10, color=GREY)

out = os.path.dirname(os.path.abspath(__file__))
for ext in ("png", "svg", "pdf"):
    fig.savefig(os.path.join(out, f"kscdiff_architecture.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(out, "kscdiff_architecture.*"))
