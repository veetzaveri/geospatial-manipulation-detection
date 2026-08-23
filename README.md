# CRP Geospatial Manipulation Benchmark Code

Code release for:

> Veet Zaveri and Arun S. Maiya, “Towards a Better Characterization of Adversarial Attacks in Geospatial Imagery,” *Remote Sensing* 18(12), 2041 (2026). https://doi.org/10.3390/rs18122041

This release implements the paper's REAL/PIXEL/PATCH/FAKE benchmark, all reported follow-on experiments, dataset-generation procedures, and statistical analyses.

## Scope and provenance

Only one original executable file was recoverable from the prior project conversations: the Swin-B main experiment script in `archive/recovered/`. A readable LF-normalized copy and a ZIP preserving the original CRLF bytes are supplied. The clean, reusable pipeline in `src/` and `scripts/` was reconstructed from that file, the published paper, and recoverable prior-chat implementation details.

The clean scripts use the published paper as their specification, including Table 2. The
historical Swin-B file is retained only as archival provenance and is not called by the clean
entry points. See [docs/PROVENANCE.md](docs/PROVENANCE.md) for the reconstruction boundary.

## Published Table 2 protocol

| Component | Main benchmark | FSC adaptation |
|---|---|---|
| Input/preprocessing | RGB 224x224, bicubic, tensor, ImageNet mean/std | Same deterministic preprocessing |
| Training augmentation | LightAug: horizontal flip p=0.5; brightness/contrast/saturation 0.05 | Same for fine-tuning; none for validation/test/zero adaptation |
| Loss | Four-way cross-entropy; binary BCE-with-logits | Same task-appropriate losses |
| Optimizer | AdamW, lr 3e-5, weight decay 1e-5 | Same |
| Batch size | Fixed by architecture in the reported 64-72 range | 32 fine-tuning; 64 zero adaptation |
| Maximum epochs | 30 | 20; zero adaptation has no training |
| Scheduler | Cosine warm restarts: T0=5, Tmult=2, eta-min=1e-6 | Same |
| Early stopping | Validation macro-F1, min delta 1e-4, patience 5 | Validation selection metric, min delta 1e-4, patience 3 |
| Decisions | Four-way argmax; binary threshold maximizes validation attack F1 | Argmax/exact-family; binary validation threshold maximizes attack F1 |
| Reproducibility | Seed 42 and deterministic CuDNN/PyTorch settings where supported | Same |

## Repository contents

| Experiment | Script(s) |
|---|---|
| Main 20-model four-way benchmark and FAKE subtype metrics | `scripts/train_main.py`, `scripts/evaluate_main.py` |
| Specialized REAL/FAKE, REAL/PIXEL, REAL/PATCH detectors | `scripts/train_binary.py` |
| FAKE leave-one-method-out source shift | `scripts/train_fake_lomo.py` |
| PIXEL B=50/B=400 and unseen square PATCH | `scripts/generate_pixel.py`, `scripts/generate_patch.py`, `scripts/evaluate_attack_variant.py` |
| No-Faster-R-CNN/random-placement PIXEL control | `scripts/generate_pixel.py --placement random`, `scripts/evaluate_pixel_no_oracle.py` |
| FSC source-ID-disjoint splits and zero adaptation | `scripts/prepare_fsc_splits.py`, `scripts/evaluate_fsc_zero_adaptation.py` |
| FSC 50/200/1000-image-per-class full fine-tuning | `scripts/finetune_fsc.py` |
| Binary vs Multi-C vs Multi-E, paired bootstrap, McNemar | `scripts/compare_binary_multi.py`, `scripts/summarize_fsc_comparisons.py` |
| Light/medium/heavy augmentation | `scripts/run_augmentation_study.py` |
| Low-prevalence triage and false alarms | `scripts/low_prevalence_triage.py` |
| FAKE-inpaint generation and feathered composite | `scripts/generate_fake_inpaint.py`, `scripts/feather_inpaint_outputs.py` |
| FAKE-GeoSynth generation | `scripts/generate_fake_geosynth.py` |
| Base split construction and final assembly | `scripts/prepare_base_splits.py`, `scripts/assemble_benchmark.py`, `scripts/validate_dataset.py` |

The complete table-to-script mapping is in [docs/EXPERIMENT_MANIFEST.md](docs/EXPERIMENT_MANIFEST.md).
The parameter-level audit is in [docs/PAPER_PROTOCOL_MAP.md](docs/PAPER_PROTOCOL_MAP.md).
Release-candidate checks and their limits are recorded in [docs/VALIDATION.md](docs/VALIDATION.md).

## Installation

The paper reports Python 3.13.2, PyTorch 2.7.0+cu126, torchvision 0.22.0+cu126, Ubuntu 22.04, and a 24 GB Quadro P6000 GPU.

```bash
conda env create -f environment.yml
conda activate crp-geospatial-benchmark
```

`environment.yml` pins the paper's `torch==2.7.0+cu126` and
`torchvision==0.22.0+cu126` wheels. `requirements.txt` provides the equivalent pip setup.

## Expected main-dataset layout

```text
DATASET_ROOT/
  train/
    REAL/
    PIXEL/
    PATCH/
    FAKE/
      FAKE-inpaint/
      FAKE-StyleGAN2/
      FAKE-GeoSynth/
  validation/
    ...same labels...
  test/
    ...same labels...
```

Validate counts and matched REAL/PIXEL/PATCH filenames:

```bash
python scripts/validate_dataset.py --dataset-root /path/to/DATASET_ROOT
```

The expected counts are 21,680/class for training and 2,710/class for validation and test. FAKE contains 10,840/1,355/1,355 inpainting images, 5,420/677/677 StyleGAN2 images, and 5,420/678/678 GeoSynth images across train/validation/test.

## Main benchmark

```bash
python scripts/train_main.py \
  --dataset-root /path/to/DATASET_ROOT \
  --model swin_b \
  --augmentation lightaug \
  --output-dir outputs/main/swin_b
```

Run the same command for the model keys listed in `configs/experiment_matrix.yaml`. The
default batch size is fixed at 72 for Swin-B and 64 for the other clean adapters, satisfying
Table 2's reported 64-72 range; `--batch-size` is available only when hardware requires an
override. GeoRSCLIP, RemoteCLIP, and SatDINO require their public foundation checkpoint via
`--pretrained-checkpoint`. SatlasPretrain uses its package-supplied weights.

## Binary specialists

```bash
python scripts/train_binary.py \
  --dataset-root /path/to/DATASET_ROOT \
  --model swin_b \
  --attack FAKE \
  --output-dir outputs/binary/swin_b/real_vs_fake
```

Repeat with `PIXEL` and `PATCH`. The final threshold is chosen on validation data to maximize attack-class F1.

## FAKE LOMO

```bash
python scripts/train_fake_lomo.py \
  --dataset-root /path/to/DATASET_ROOT \
  --model swin_b \
  --held-out FAKE-inpaint \
  --output-dir outputs/lomo/swin_b/holdout_inpaint
```

Repeat for `FAKE-GeoSynth` and `FAKE-StyleGAN2`, and for the five representative architectures. REAL, PIXEL, and PATCH remain unchanged; only the held-out FAKE subtype is removed from train/validation.

## PIXEL/PATCH variants

Generate the paper's main PIXEL setting with `--budget 200`, and stress tests with `--budget 50` or `--budget 400`:

```bash
python scripts/generate_pixel.py \
  --input-root /path/to/split/REAL \
  --output-root /path/to/variant/PIXEL \
  --placement oracle \
  --budget 50
```

The no-oracle control changes only `--placement random`. Generate the square-PATCH stress set with:

```bash
python scripts/generate_patch.py \
  --input-root /path/to/split/REAL \
  --output-root /path/to/variant/PATCH \
  --geometry square --radius 5
```

Then evaluate a balanced variant tree using `evaluate_attack_variant.py`, or a PIXEL-only no-oracle tree using `evaluate_pixel_no_oracle.py`.

## FSC transfer and adaptation

FSC manifests use columns `path,label,source_id,split`. All variants sharing a source image ID must remain in the same split.

```bash
python scripts/prepare_fsc_splits.py \
  --inventory /path/to/fsc_inventory.csv \
  --output-manifest /path/to/fsc_manifest.csv

python scripts/finetune_fsc.py \
  --manifest /path/to/fsc_manifest.csv \
  --mode multi \
  --shots-per-class 200 \
  --model swin_b \
  --in-domain-checkpoint outputs/main/swin_b/best_checkpoint.pth \
  --output-dir outputs/fsc/swin_b/multi/shot_200
```

Use `--mode binary --attack FAKE|PIXEL|PATCH` for specialists. The code fully fine-tunes
backbone and head parameters for at most 20 epochs with Table 2 early stopping.

For binary zero-adaptation evaluation, reuse the threshold selected on the original in-domain
validation split; FSC labels are not used:

```bash
python scripts/evaluate_fsc_zero_adaptation.py \
  --manifest /path/to/fsc_zero_adaptation_manifest.csv \
  --mode binary --attack FAKE \
  --model swin_b \
  --checkpoint outputs/binary/swin_b/real_vs_fake/best_checkpoint.pth \
  --threshold-json outputs/binary/swin_b/real_vs_fake/validation_threshold.json \
  --output-dir outputs/fsc/swin_b/zero/real_vs_fake
```

`compare_binary_multi.py --setting in-domain` uses Multi-C hard decisions (four-way argmax
collapsed to REAL/non-REAL) and ordinary paired image-level bootstrap. `--setting fsc` uses
Multi-E hard decisions (argmax must equal the evaluated attack family) and paired stratified
bootstrap, with 10,000 replicates for BA/F1 and 1,000 for AUPRC.

## Low-prevalence triage

```bash
python scripts/low_prevalence_triage.py \
  --predictions outputs/main/swin_b/test/predictions.csv \
  --output-json outputs/triage/swin_b.json
```

The implementation uses `1-p(REAL)`, simulated prevalence 1%, 5%, and 10%, a 90%-recall operating point, 2,000 ordinary image-level bootstrap replicates, and the exact one-sided upper bound when zero false positives are observed.

## Outputs

Training scripts retain:

- `best_checkpoint.pth` and `run_metadata.json`;
- complete epoch history;
- per-image paths, labels, predictions, and class probabilities;
- confusion matrices and aggregate metrics;
- FAKE-subtype metrics where applicable;
- bootstrap configuration, estimates, and intervals.

Absolute data and checkpoint paths are command-line inputs and are not embedded in the clean source.

## License and citation

An MIT license is supplied as a conventional software-release default. `CITATION.cff` cites both this software and the associated article.
