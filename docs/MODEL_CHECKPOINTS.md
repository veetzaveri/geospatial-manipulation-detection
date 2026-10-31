# Models and checkpoint inputs

The registry in `crp_benchmark.models.MODEL_SPECS` exposes all 20 architectures in Table 1.

## Model identifiers

- Swin-B: `swin_base_patch4_window7_224.ms_in22k`.
- EfficientNetV2-S: `tf_efficientnetv2_s.in21k_ft_in1k`, with the legacy timm alias `tf_efficientnetv2_s_in21ft1k` as a fallback.
- Satlas-Swin-B: `Aerial_SwinB_SI`, `fpn=True`, classification head.
- SatDINO-B/16: `crp_benchmark.satdino.SatDINOEncoder`, with the GSD register and 198-token positional embedding.
- Remaining identifiers are specified in `MODEL_SPECS`.

All models use the 224x224 RGB preprocessing and ImageNet normalization in Table 2.

## Foundation weights for initial training

GeoRSCLIP-B/32, RemoteCLIP-B/32, and SatDINO-B/16 accept foundation weights through `--pretrained-checkpoint`.

- [GeoRSCLIP / RS5M](https://github.com/om-ai-lab/RS5M)
- [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP)
- [SatDINO](https://github.com/strakaj/SatDINO)
- [SatlasPretrain](https://github.com/allenai/satlaspretrain_models)

Use the official SatDINO ViT-B/16 teacher-backbone checkpoint. Loading is strict, including the GSD register. A standard ViT checkpoint has a different architecture.

Satlas uses its upstream package model-construction workflow. Standard timm models obtain their foundation weights through timm.

## Experiment checkpoints

Training saves `best_checkpoint.pth` containing the full model state and run metadata. Evaluation and FSC adaptation restore the corresponding checkpoint with strict state-dictionary loading.

GeoRSCLIP, RemoteCLIP, and SatDINO experiment checkpoints contain the encoder weights, so those models do not need a separate foundation checkpoint when restoring a complete experiment checkpoint.

For FSC adaptation, use the matching in-domain model and task checkpoint. Binary zero-adaptation evaluation additionally uses the in-domain `validation_threshold.json`; adapted binary comparisons use the threshold selected on the FSC validation split.

## Release scope

This repository distributes code. Datasets and weight files are supplied separately by the person running the experiments, subject to their source licenses and access terms.
