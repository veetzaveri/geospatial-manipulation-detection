# Paper protocol map

## Table 2 implementation

| Published component | Code location |
|---|---|
| RGB 224x224, bicubic resize, tensor conversion, ImageNet mean/std | `crp_benchmark.data.build_transform` |
| LightAug and deterministic validation/test transforms | `crp_benchmark.data.build_transform` |
| Four-way cross-entropy and binary BCE-with-logits | `crp_benchmark.training.fit_model` |
| AdamW, lr 3e-5, weight decay 1e-5 | `crp_benchmark.training.fit_model` |
| Main batch size 64-72, fixed by architecture | `crp_benchmark.models.paper_main_batch_size` |
| FSC batch size 32; zero-adaptation batch size 64 | `finetune_fsc.py`; `evaluate_fsc_zero_adaptation.py` |
| Main maximum 30 epochs, validation macro-F1, delta 1e-4, patience 5 | `crp_benchmark.training.MAIN_PROTOCOL` |
| FSC maximum 20 epochs, validation selection metric, delta 1e-4, patience 3 | `crp_benchmark.training.FSC_PROTOCOL` |
| Cosine warm restarts, T0=5, Tmult=2, eta-min=1e-6 | `crp_benchmark.training.fit_model` |
| Four-way argmax and validation-selected binary attack-F1 threshold | evaluation/training entry points |
| Seed 42 across Python, NumPy, PyTorch, CUDA, workers, sampling and splits | `crp_benchmark.reproducibility`; split scripts |
| Deterministic CuDNN/PyTorch settings and runtime record | `crp_benchmark.reproducibility`; `run_metadata.json` |
| Reported Python/PyTorch/torchvision/CUDA wheel versions | `environment.yml`; `requirements.txt` |

## Experiment-specific implementation

| Paper results | Protocol encoded by the scripts |
|---|---|
| Tables 3-5 | Twenty-model four-way training; ordinary 10,000-replicate image-level bootstrap; FAKE recall CIs resampled within each subtype |
| Tables 6-7 | One FAKE subtype excluded from train/validation; mixed and excluded-subtype tests; 10,000-replicate nonparametric bootstrap |
| Tables 8-9 | Original four-way checkpoint; balanced 10,840-image B=50, B=400 and square-PATCH sets; target recall and missed-as-REAL diagnostics |
| Table 10 | Original four-way checkpoint; 2,710 random-placement/no-oracle PIXEL examples; PIXEL recall, REAL/other misses and mean PIXEL probability |
| Table 11 | Exact matched REAL/attack images; binary validation threshold; Multi-C hard decision is four-way argmax collapsed to non-REAL; ordinary paired 10,000-replicate bootstrap and paired McNemar test |
| Tables 12-14 | No FSC labels used for adaptation; binary threshold carried from in-domain validation; Multi-E requires exact attack-family argmax; FSC paired stratified bootstrap (10,000 BA/F1, 1,000 AUPRC) |
| Tables 15-21 | Source-ID-disjoint FSC; 542/class validation and test; 50/200/1000 shots/class; full-model fine-tuning; Multi-E detailed comparisons and paired model-attack-unit summary deltas |
| Table 22 | Five representative models crossed with LightAug, MedAug and HeavyAug; augmentation only in training |
| Table 23 | Score `1-p(REAL)`; 1%, 5% and 10% simulated prevalence; approximately 90% recall; prevalence-adjusted AUPRC/precision; ordinary 2,000-replicate bootstrap with the operating point recomputed |

## Generation procedures

- PIXEL implements the paper's frozen Faster R-CNN ResNet-50-FPN saliency placement,
  B=200 main setting, B=50/400 stress settings, 5x5 neighborhoods, epsilon 0.05, dummy
  weight 0.05, box restriction, and random-placement control.
- PATCH implements circular radius-5 training geometry, square radius-5 stress geometry,
  momentum 0.75 and 50 iterations. The paper does not state the exact patch count, step size,
  TV weight, early-stop confidence or detector box threshold; those are explicit CLI options.
- FAKE-inpaint implements the reported 512x512 canvas, deterministic 2-3 rectangle/ellipse
  masks, 0.18-0.35 extent, 35% cap, 15 steps, guidance 7.5, strength 0.95 and 8-pixel
  feathered composite. The exact original prompt bank/model ID is not stated in the paper.
- FAKE-GeoSynth implements the plain text-to-image pipeline, 512x512 generation, 15 steps,
  guidance 5.5 and bottom crop. The exact prompt bank and crop depth are not stated in the
  paper and remain explicit inputs.
