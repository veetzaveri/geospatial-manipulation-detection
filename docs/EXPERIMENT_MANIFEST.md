# Published experiment-to-code manifest

| Paper result | Models | Public-release entry point | Provenance |
|---|---:|---|---|
| Tables 3-4: main four-way accuracy/per-class F1 | 20 | `train_main.py`, `evaluate_main.py` | Swin-B exact template; other adapters reconstructed |
| Table 5: matched FAKE subtype recall | 20 | automatic output of main evaluation | exact metric logic recovered from Swin-B |
| Tables 6-7: FAKE LOMO | representative 5 | `train_fake_lomo.py` | protocol/filenames recovered; implementation reconstructed |
| Tables 8-9: PIXEL B=50/B=400, square PATCH | representative 5 | `generate_pixel.py`, `generate_patch.py`, `evaluate_attack_variant.py` | paper reconstruction |
| Table 10: no-oracle PIXEL | representative 5 | `generate_pixel.py --placement random`, `evaluate_pixel_no_oracle.py` | evaluator filenames recovered; implementation reconstructed |
| Table 11: in-domain binary vs Multi-C | representative 5 x 3 pairs | `train_binary.py`, `compare_binary_multi.py --setting in-domain` | four-way argmax collapsed to REAL/non-REAL; ordinary paired bootstrap |
| Tables 12-14: FSC zero adaptation | representative 5 x 3 pairs | `evaluate_fsc_zero_adaptation.py`, `compare_binary_multi.py --setting fsc`, `summarize_fsc_comparisons.py` | in-domain binary threshold; exact-family Multi-E; paired stratified bootstrap |
| Tables 15-21: FSC 50/200/1000 adaptation | representative 5 x 3 pairs | `finetune_fsc.py`, `compare_binary_multi.py --setting fsc`, summary script | full fine-tuning; exact-family Multi-E; paired stratified bootstrap |
| Table 22: augmentation severity | representative 5 x 3 regimes | `run_augmentation_study.py` | filenames and transforms recovered |
| Table 23: low-prevalence triage | representative 5; three settings | `low_prevalence_triage.py` | equations/replicates paper-specified |

Grad-CAM does not appear in the final published paper and is intentionally not represented as a published experiment.
