"""Capture the intermediate quantities of one RA-PhyKSC-Diff iteration for Fig. 4.

    python figures/capture_ra_iteration.py [IDX] [T_STOP]  ->  figures/ra_iteration.npz
Runs the fixed-step PhyKSC-Diff trajectory down to level T_STOP, then records one iteration
(prediction, FFT, first DC, IFFT, residual, jump, second DC) exactly as in ksc/diffusion.py.
"""
import os
import sys

import numpy as np
import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))
from evaluate import load                              # noqa: E402
from ksc.data import KneeSinglecoil, fft2c, ifft2c    # noqa: E402

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 280
t_stop = int(sys.argv[2]) if len(sys.argv) > 2 else 500
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = torch.device("cuda")
ds = KneeSinglecoil(os.path.join(here, "..", "cache/fastmri_mc_vc_test.npz"), acceleration=4,
                    center_fraction=0.08, fixed_mask=True, seed=1234)
phy = load(os.path.join(here, "..", "runs/phyksc/last.pt"), dev)
k, mask, order, _ = ds[idx]
k, mask, order = k[None].to(dev), mask[None].to(dev), order[None].to(dev)
y = k * mask[:, None, :]

with torch.no_grad():
    # fixed-step trajectory from t = 1000 down to t_stop (same update as the sampler)
    x = ifft2c(y)
    prev_R = None
    for t in range(1000, t_stop, -1):
        tt = torch.tensor([t], device=dev)
        k0 = phy.dc(fft2c(phy.predict(x, tt)), y, mask)
        prev_R = ((ifft2c(k0) - x).norm() / x.norm()).item()
        x_next = x - ifft2c(phy.degrade(k0, order, mask, tt)) + ifft2c(phy.degrade(k0, order, mask, tt - 1))
        x = ifft2c(phy.dc(fft2c(x_next), y, mask))
    # one recorded iteration at t = t_stop
    tt = torch.tensor([t_stop], device=dev)
    x_hat = phy.predict(x, tt)
    k_hat = fft2c(x_hat)
    k_dc = phy.dc(k_hat, y, mask)
    x_dc = ifft2c(k_dc)
    R = ((x_dc - x).norm() / x.norm()).item()
    dR = abs(R - prev_R) / prev_R
    jump = 1 if dR > 0.02 else (5 if dR > 0.005 else 10)
    t_next = torch.tensor([max(0, t_stop - jump)], device=dev)
    x_jump = x - ifft2c(phy.degrade(k_dc, order, mask, tt)) + ifft2c(phy.degrade(k_dc, order, mask, t_next))
    k_next = phy.dc(fft2c(x_jump), y, mask)
    x_next = ifft2c(k_next)

to = lambda a: a[0].cpu().numpy()
np.savez_compressed(os.path.join(here, "ra_iteration.npz"),
                    x_old=np.abs(to(x)), x_hat=np.abs(to(x_hat)), k_hat=np.abs(to(k_hat)), y=np.abs(to(y)),
                    mask=np.repeat(to(mask)[None, :], 320, 0), x_dc=np.abs(to(x_dc)), x_next=np.abs(to(x_next)),
                    retained=np.repeat(to(phy.degrade_mask(order, mask, tt))[0][None, :], 320, 0),
                    t=t_stop, t_next=int(t_next), R=R, dR=dR, jump=jump)
print(f"slice {idx}: t={t_stop} R={R:.4f} dR={dR:.4f} -> dt={jump}, t_next={int(t_next)}")
