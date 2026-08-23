# Model and checkpoint notes

The clean registry exposes all 20 paper architecture keys through `crp_benchmark.models.MODEL_SPECS`.

## Recovered identifiers

- Swin-B: `swin_base_patch4_window7_224.ms_in22k`.
- EfficientNetV2-S: prior code used the legacy timm identifier `tf_efficientnetv2_s_in21ft1k`; the clean registry first tries the current dotted alias and then the legacy alias.
- Satlas-Swin-B: `Aerial_SwinB_SI`, `fpn=True`, classification head.

## External checkpoint arguments

GeoRSCLIP-B/32, RemoteCLIP-B/32, and SatDINO-B/16 require `--pretrained-checkpoint`. Foundation weights are not redistributed here. Verify the original sources and licenses before downloading or releasing derived checkpoints.

- GeoRSCLIP/RS5M: <https://github.com/om-ai-lab/RS5M>
- RemoteCLIP: <https://github.com/ChenDelong1999/RemoteCLIP>
- SatDINO: <https://github.com/strakaj/SatDINO>
- SatlasPretrain models: <https://github.com/allenai/satlaspretrain_models>

## Reconstructed standard identifiers

The remaining timm identifiers implement the architecture/pretraining descriptions in Table 1, but the original per-model source files were not recoverable. Package/version aliases can change. Record the resolved model identifier and package version in every run metadata file, then compare performance with the published values before treating a rerun as equivalent.

## Satlas preprocessing

The clean `satlas_swin_b` entry point follows published Table 2: bicubic resize to 224x224
and ImageNet normalization. This intentionally makes the paper, rather than an incomplete
historical reconstruction, the runnable protocol specification.
