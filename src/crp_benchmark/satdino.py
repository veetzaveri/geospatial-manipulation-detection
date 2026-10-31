"""SatDINO ViT-B/16 encoder for the paper's fixed 224x224 RGB protocol.

Uses timm's ViT blocks with SatDINO's additional GSD register and original
token/position order: [CLS, image patches, GSD]. Parameter names and shapes
match the official SatDINO teacher backbone (not an ordinary timm ViT).

Architecture reference: https://github.com/strakaj/SatDINO
"""
from __future__ import annotations

from functools import partial

import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn


class SatDINOEncoder(VisionTransformer):
    def __init__(self) -> None:
        super().__init__(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=0,
            global_pool="token",
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            reg_tokens=1,
            no_embed_class=False,
            fc_norm=False,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        # Retain the official checkpoint key rather than timm's reg_token key.
        self.gsd_register = self.reg_token
        self.register_parameter("reg_token", None)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.patch_embed.num_patches:
            raise ValueError("SatDINOEncoder expects RGB images resized to 224x224")
        batch_size = x.shape[0]
        tokens = torch.cat(
            (
                self.cls_token.expand(batch_size, -1, -1),
                x,
                self.gsd_register.expand(batch_size, -1, -1),
            ),
            dim=1,
        )
        return self.pos_drop(tokens + self.pos_embed)
