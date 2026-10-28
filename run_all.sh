#!/usr/bin/env bash
# RQ1-RQ3 pipeline on the fastMRI multicoil knee + brain volumes (emulated single-coil):
#   preprocess -> train KSC-Diff and PhyKSC-Diff (50 epochs each) -> evaluate -> report.
# Usage: ./run_all.sh [results_dir]     (default results/run; results/main holds the published results)
# Each stage is skipped if its output already exists, so the script can be rerun to resume.
set -eo pipefail
cd "$(dirname "$0")"
OUT="${1:-results/run}"
mkdir -p runs/ksc runs/phyksc "$OUT"

# preprocessing: reuse the published train/test split when it is present
if [ ! -f cache/fastmri_mc_vc_test.npz ]; then
    if [ -f cache/fastmri_mc_vc_split.json ]; then
        python preprocess.py --split cache/fastmri_mc_vc_split.json
    else
        python preprocess.py
    fi
fi

# training (a model counts as trained only if its 50-epoch log AND checkpoint exist)
trained() { [ -f "$1/last.pt" ] && [ "$(tail -n +2 "$1/train_log.csv" | wc -l)" -ge 50 ]; }
trained runs/ksc || python train.py --epochs 50 --snapshot-epochs 20 --out runs/ksc 2>&1 | tee -a runs/ksc/train_stdout.log
trained runs/phyksc || python train.py --dc-train --epochs 50 --snapshot-epochs 20 --out runs/phyksc 2>&1 | tee -a runs/phyksc/train_stdout.log

# quick GPU smoke test of every evaluation path before the long run
python evaluate.py --out results/smoke --main-T 20 --T-list 10 --limit 2 --ablation-per-volume 30 --probe-per-volume 30
rm -rf results/smoke
python evaluate.py --out "$OUT" 2>&1 | tee -a "$OUT/eval_stdout.log"
python report.py --res "$OUT"
python paper_numbers.py "$OUT"
