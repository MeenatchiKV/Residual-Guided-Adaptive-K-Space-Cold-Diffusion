"""Reconstruct one test slice with KSC-Diff, PhyKSC-Diff and RA-PhyKSC-Diff (T=1000) and save it
for the qualitative figure:  python figures/recon_slice.py IDX  ->  results/main/recon/sliceIDX.npz"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from evaluate import load                      # noqa: E402
from ksc.data import KneeSinglecoil, ifft2c   # noqa: E402

idx = int(sys.argv[1])
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
torch.backends.cudnn.benchmark = True          # same numerics as evaluate.py
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = torch.device("cuda")
ds = KneeSinglecoil(os.path.join(root, "cache/fastmri_mc_vc_test.npz"), acceleration=4,
                    center_fraction=0.08, fixed_mask=True, seed=1234)   # same masks as evaluate.py
ksc = load(os.path.join(root, "runs/ksc/last.pt"), dev)
phy = load(os.path.join(root, "runs/phyksc/last.pt"), dev)
k, mask, order, _ = ds[idx]
k, mask, order = k[None].to(dev), mask[None].to(dev), order[None].to(dev)
out = {"gt": ifft2c(k)[0].abs().cpu().numpy(),
       "Zero-filled@0": ifft2c(k * mask[:, None, :])[0].abs().cpu().numpy()}
for name, diff, kw in [("KSC-Diff@1000", ksc, dict(physics=False)),
                       ("PhyKSC-Diff@1000", phy, dict(physics=True)),
                       ("RA-PhyKSC-Diff@1000", phy, dict(physics=True, adaptive=True))]:
    img, nfe, _, _ = diff.sample(k, mask, order, steps=1000, **kw)
    out[name] = img[0].abs().cpu().numpy()
    print(name, "NFE", int(nfe[0]))
np.savez_compressed(os.path.join(root, "results/main/recon", f"slice{idx:03d}.npz"), **out)
print("saved", ds.files[idx], "slice", int(ds.slices[idx]))
