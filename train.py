"""Train a k-space cold-diffusion denoiser (L1 x0-prediction, T=1000).

  python train.py --out runs/ksc                 KSC-Diff    (no data consistency)
  python train.py --out runs/phyksc --dc-train   PhyKSC-Diff (hard DC layer in the training loss)

RA-PhyKSC-Diff is training-free and reuses the PhyKSC-Diff checkpoint.
"""
import argparse
import copy
import csv
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ksc.data import KneeSinglecoil
from ksc.diffusion import KSpaceColdDiffusion
from ksc.unet import Unet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="cache/fastmri_mc_vc_train.npz", help="cache built by preprocess.py")
    p.add_argument("--out", default="runs/ksc")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=6, help="effective batch size")
    p.add_argument("--accum", type=int, default=2, help="gradient accumulation (micro-batch = batch/accum)")
    p.add_argument("--snapshot-epochs", type=int, nargs="*", default=[20])
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--max-steps", type=int, default=0, help="stop after N optimiser steps (smoke test)")
    p.add_argument("--dc-train", action="store_true", help="PhyKSC-Diff: DC layer in the training loss")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)

    ds = KneeSinglecoil(args.data, acceleration=4, center_fraction=0.08)
    dl = DataLoader(ds, batch_size=args.batch_size // args.accum, shuffle=True, drop_last=True, num_workers=0)
    print(f"train slices: {len(ds)}  optimiser steps/epoch: {len(dl) // args.accum}")

    model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=2).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    diff = KSpaceColdDiffusion(model, args.timesteps)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    ckpt_path = os.path.join(args.out, "last.pt")
    start_epoch, step = 1, 0
    if os.path.exists(ckpt_path) and not args.max_steps:
        ck = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); start_epoch, step = ck["epoch"] + 1, ck["step"]
        print(f"resumed from epoch {ck['epoch']}")

    log_path = os.path.join(args.out, "train_log.csv")
    new_log = not os.path.exists(log_path) or args.max_steps
    log = open(log_path, "w" if new_log else "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["epoch", "step", "loss", "seconds"])

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for i, (k, mask, order, _) in enumerate(dl):
            k, mask, order = k.to(dev), mask.to(dev), order.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = diff.loss(k, mask, order, dc_train=args.dc_train)
            (loss / args.accum).backward()
            losses.append(loss.item())
            if (i + 1) % args.accum:
                continue
            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - args.ema_decay)
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {np.mean(losses[-100:]):.5f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        secs = time.time() - t0
        writer.writerow([epoch, step, float(np.mean(losses)), round(secs, 1)]); log.flush()
        print(f"== epoch {epoch} done: loss {np.mean(losses):.5f}  {secs / 60:.1f} min", flush=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "step": step, "args": vars(args)}, ckpt_path)
        if epoch in args.snapshot_epochs:
            torch.save({"ema": ema.state_dict(), "epoch": epoch, "step": step},
                       os.path.join(args.out, f"ema_epoch{epoch}.pt"))
        if args.max_steps and step >= args.max_steps:
            break
    log.close()


if __name__ == "__main__":
    main()
