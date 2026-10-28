# Residual-Guided Adaptive K-Space Cold Diffusion for Accelerated MRI Reconstruction

Code, results and figures for *Residual-Guided Adaptive K-Space Cold Diffusion for
Physics-Consistent Accelerated MRI Reconstruction* (Meenatchi K V and Sountharrajan S,
Amrita Vishwa Vidyapeetham, Chennai).

The study builds three systematically related k-space cold-diffusion models and evaluates them on
emulated single-coil fastMRI knee and brain data with R4 Cartesian undersampling:

| Model | Training | Inference |
|---|---|---|
| **KSC-Diff** | L1 loss on the predicted clean image, x̂₀ = f_θ(x_t, t), t ~ U{0, …, 1000} | cold-diffusion reverse process, Δt = 1 |
| **PhyKSC-Diff** | same loss on the data-consistent prediction, k̂ ← M⊙y + (1−M)⊙k̂ | Δt = 1, DC layer kept in the forward pass |
| **RA-PhyKSC-Diff** | none (reuses the PhyKSC-Diff network) | DC + residual-guided step size Δt ∈ {1, 5, 10} |

RA-PhyKSC-Diff computes the reconstruction-change residual R_i = ‖x_i^DC − x^(i)‖ / ‖x^(i)‖ and its
relative change ΔR_i = |R_i − R_{i−1}| / R_{i−1}, and selects Δt = 1 / 5 / 10 for ΔR_i > 0.02 /
(0.005, 0.02] / ≤ 0.005.

## Main results (591 test slices, R4)

| | NMSE ↓ | PSNR (dB) ↑ | SSIM ↑ | NFE ↓ | Median time / slice |
|---|---|---|---|---|---|
| Zero-filled | 0.039060 | 27.36 | 0.6579 | 0 | – |
| KSC-Diff | 0.024900 | 29.57 | 0.7087 | 1000 | 52.7 s |
| PhyKSC-Diff | 0.024051 | 29.73 | 0.7110 | 1000 | 53.0 s |
| RA-PhyKSC-Diff | 0.024022 | 29.74 | 0.7111 | 186.2 | 10.2 s |

* **RQ1:** data-consistent training improves all metrics (PSNR +0.16 dB, NMSE −3.41 %, SSIM +0.0022;
  better PSNR on 89.5 % of slices; Holm-adjusted p < 10⁻⁶⁰).
* **RQ3:** RA-PhyKSC-Diff uses 81.4 % fewer network evaluations and satisfies a predefined tolerance
  (PSNR −0.1 dB, SSIM −0.005, NMSE +2 %) on every test slice.
* **RQ2:** matched-NFE controls (uniform and randomly ordered step schedules) perform equally well, and
  ΔR_i does not predict the error of skipping steps: the saving comes from shortening the trajectory,
  not from the residual timing the transitions.

All numbers in the paper are in [`results/main/paper_numbers.json`](results/main/paper_numbers.json);
per-slice results are in [`results/main/per_slice.csv`](results/main/per_slice.csv).

## Repository layout

```
preprocess.py        multicoil knee + brain -> emulated single-coil slices (cache/*.npz), train/test split
ksc/data.py          dataset, R4 random column mask (centre fraction 0.08), normalization
ksc/unet.py          time-conditioned ConvNeXt U-Net, 64-128-256-512, linear attention (56.6 M parameters)
ksc/diffusion.py     degradation D_t, training losses, KSC / PhyKSC / RA-PhyKSC samplers
train.py             trains KSC-Diff (default) or PhyKSC-Diff (--dc-train)
evaluate.py          Experiments I-III on the test slices -> <results>/per_slice.csv, probe.csv
report.py            tables, Wilcoxon / Holm and non-inferiority tests -> <results>/RESULTS.md, table*.csv
paper_numbers.py     every number reported in the paper -> <results>/paper_numbers.json
run_all.sh           preprocess -> train both models -> evaluate -> report (resumable)
figures/             scripts and files for all paper figures
cache/fastmri_mc_vc_split.json   the 100 fastMRI volumes used and their train/test assignment
results/main/        published results (per-slice metrics, tables, trajectories, figures)
runs/*/train_log.csv training curves
```

## Setup

Python 3.11 with a CUDA GPU (the experiments used an RTX 4050 Laptop GPU, 6 GB).

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data

The fastMRI data cannot be redistributed. Request access at <https://fastmri.med.nyu.edu/> and place
the multicoil training volumes in

```
downloads/knee_multicoil_train/*.h5
downloads/brain_multicoil_train/*.h5
```

Only the 100 volumes listed in `cache/fastmri_mc_vc_split.json` (50 knee, 50 brain) are used; other
files in these folders are ignored. Preprocessing emulates single-coil data with least-squares coil
weights fitted to the root-sum-of-squares image (Tygert & Zbontar, 2020), crops/pads to 320×320 and
skips 5 (knee) or 2 (brain) boundary slices per side, giving 1,374 training and 591 test slices.

## Reproduce

```bash
./run_all.sh            # writes to results/run; results/main keeps the published results
```

or step by step:

```bash
python preprocess.py --split cache/fastmri_mc_vc_split.json
python train.py --epochs 50 --snapshot-epochs 20 --out runs/ksc
python train.py --dc-train --epochs 50 --snapshot-epochs 20 --out runs/phyksc
python evaluate.py --out results/run
python report.py --res results/run
python paper_numbers.py results/run
```

Training takes about 4 h per model and the full evaluation about 25 h on the laptop GPU above.
`train.py` and `evaluate.py` resume where they stopped. Reconstructions are reproducible within a run;
between runs, GPU kernel autotuning can change individual step decisions of RA-PhyKSC-Diff
(NFE within about 5 %, PSNR within 0.01 dB).

Trained weights (EMA, epoch 50) are attached to the repository's GitHub release.

## Protocol details

* **Test inputs:** the R4 mask and the degradation order are seeded per slice, so every method sees the
  same input.
* **Diffusion-time configuration T:** the reverse trajectory always starts from t = 1000 (the
  undersampled measurement) and visits T evenly spaced levels, so a fixed-step sampler uses NFE = T.
  Adaptive jumps are counted in levels of that grid.
* **Evaluation subsets:** the T = 1000 comparisons use all 591 test slices; the T = 20–500 ablation,
  matched-NFE controls and threshold sweep use 5 evenly spaced slices per test volume (150); the
  residual–error analysis uses 2 slices per volume (60).
* **Metrics:** magnitude images; NMSE = ‖x − x̂‖² / ‖x‖²; PSNR and SSIM with the ground-truth maximum as
  data range. Times are wall-clock per slice at batch size 1 and are reported as medians.
* **Statistics:** paired two-sided Wilcoxon signed-rank tests with Holm correction; one-sided
  non-inferiority Wilcoxon tests with margins PSNR 0.1 dB, SSIM 0.005, NMSE +2 % (all three must pass).

## License

MIT, see [LICENSE](LICENSE). The fastMRI data are subject to their own data-use agreement.
