"""Time-conditioned ConvNeXt U-Net (64-128-256-512, linear attention), as in KSC-Diff."""
import math

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        emb = t.float()[:, None] * freq[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class ConvNextBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim)) if time_emb_dim else None
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, t=None):
        h = self.ds_conv(x)
        if self.mlp is not None:
            h = h + self.mlp(t)[:, :, None, None]
        return self.net(h) + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden = dim_head * heads
        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = (z.reshape(b, self.heads, -1, h * w) for z in (q, k, v))
        q = q * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q).reshape(b, -1, h, w)
        return self.to_out(out) + x


class Unet(nn.Module):
    def __init__(self, dim=64, dim_mults=(1, 2, 4, 8), channels=2):
        super().__init__()
        dims = [channels, *(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

        self.downs = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(in_out):
            last = i == len(in_out) - 1
            self.downs.append(nn.ModuleList([
                ConvNextBlock(d_in, d_out, dim, norm=i != 0),
                ConvNextBlock(d_out, d_out, dim),
                LinearAttention(d_out),
                nn.Conv2d(d_out, d_out, 4, 2, 1) if not last else nn.Identity(),
            ]))
        mid = dims[-1]
        self.mid1 = ConvNextBlock(mid, mid, dim)
        self.mid_attn = LinearAttention(mid)
        self.mid2 = ConvNextBlock(mid, mid, dim)

        self.ups = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(nn.ModuleList([
                ConvNextBlock(d_out * 2, d_in, dim),
                ConvNextBlock(d_in, d_in, dim),
                LinearAttention(d_in),
                nn.ConvTranspose2d(d_in, d_in, 4, 2, 1),
            ]))
        self.final = nn.Sequential(ConvNextBlock(dim, dim), nn.Conv2d(dim, channels, 1))

    def forward(self, x, t):
        t = self.time_mlp(t)
        skips = []
        for b1, b2, attn, down in self.downs:
            x = attn(b2(b1(x, t), t))
            skips.append(x)
            x = down(x)
        x = self.mid2(self.mid_attn(self.mid1(x, t)), t)
        for b1, b2, attn, up in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = up(attn(b2(b1(x, t), t)))
        return self.final(x)
