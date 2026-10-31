# Published experiment-to-code manifest

Entry points are located in `scripts/`.

| Paper result | Models | Entry points | Evaluation |
|---|---:|---|---|
| Tables 3-4: main four-way accuracy/per-class F1 | 20 | `train_main.py`, `evaluate_main.py` | Four-way argmax; image-level bootstrap CIs |
| Table 5: FAKE-subtype recall | 20 | Automatic output of main evaluation | Recall and within-subtype bootstrap CIs |
| Tables 6-7: FAKE LOMO | Representative 5 | `train_fake_lomo.py` | Mixed-FAKE and held-out-subtype recall and diagnostics |
| Tables 8-9: PIXEL B=50/B=400, square PATCH | Representative 5 | `generate_pixel.py`, `generate_patch.py`, `evaluate_attack_variant.py` | Zero-adaptation stress tests and target-class diagnostics |
| Table 10: no-oracle PIXEL | Representative 5 | `generate_pixel.py --placement random`, `evaluate_pixel_no_oracle.py` | Random-placement PIXEL recall and missed detections |
| Table 11: in-domain binary vs Multi-C | Representative 5 x 3 pairs | `train_binary.py`, `compare_binary_multi.py --setting in-domain` | Collapsed REAL/non-REAL scoring; ordinary paired bootstrap |
| Tables 12-14: FSC zero adaptation | Representative 5 x 3 pairs | `evaluate_fsc_zero_adaptation.py`, `compare_binary_multi.py --setting fsc`, `summarize_fsc_comparisons.py` | In-domain binary threshold; Multi-E exact-label scoring; paired stratified bootstrap |
| Tables 15-21: FSC 50/200/1000 adaptation | Representative 5 x 3 pairs | `finetune_fsc.py`, comparison and summary scripts | Full fine-tuning; fixed held-out FSC splits |
| Table 22: augmentation severity | Representative 5 x 3 regimes | `run_augmentation_study.py` | LightAug, MedAug, and HeavyAug; deterministic evaluation |
| Table 23: low-prevalence triage | Representative 5; three settings | `low_prevalence_triage.py` | Prevalence-adjusted AUPRC/precision and false-alarm burden |

The five representative models are Swin-B, EfficientNet-B5, Swin-T, Satlas-Swin-B, and GeoRSCLIP-B/32.
