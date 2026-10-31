from __future__ import annotations

import torch
import pytest

from crp_benchmark.models import create_model, load_experiment_checkpoint
from crp_benchmark.satdino import SatDINOEncoder


def test_satdino_has_official_register_and_position_shapes():
    with torch.device("meta"):
        encoder = SatDINOEncoder()
    state = encoder.state_dict()
    assert state["gsd_register"].shape == (1, 1, 768)
    assert state["pos_embed"].shape == (1, 198, 768)
    assert "reg_token" not in state
    assert len(encoder.blocks) == 12
    assert encoder.num_features == 768


def test_satdino_token_order_matches_official_encoder():
    # Only positional embedding is exercised, so avoid allocating ViT weights.
    with torch.device("meta"):
        encoder = SatDINOEncoder()
    encoder.cls_token = torch.nn.Parameter(torch.full((1, 1, 768), 1.0))
    encoder.gsd_register = torch.nn.Parameter(torch.full((1, 1, 768), 3.0))
    encoder.pos_embed = torch.nn.Parameter(torch.zeros((1, 198, 768)))
    tokens = encoder._pos_embed(torch.full((2, 196, 768), 2.0))
    assert tokens.shape == (2, 198, 768)
    assert torch.all(tokens[:, 0] == 1)
    assert torch.all(tokens[:, 1:-1] == 2)
    assert torch.all(tokens[:, -1] == 3)


def test_satdino_experiment_restore_does_not_need_foundation_checkpoint(tmp_path):
    # Meta tensors check full state-dict compatibility without a large weight file.
    with torch.device("meta"):
        source, _ = create_model("satdino_b16", 4, pretrained=False)
        restored, _ = create_model("satdino_b16", 4, pretrained=False)
    checkpoint = tmp_path / "satdino_experiment.pth"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint)
    load_experiment_checkpoint(restored, checkpoint, strict=True)
    assert restored.head.out_features == 4


def test_satdino_loads_official_teacher_checkpoint_layout(tmp_path):
    with torch.device("meta"):
        backbone = SatDINOEncoder()
    checkpoint = tmp_path / "teacher.pth"
    torch.save({"teacher": backbone.state_dict()}, checkpoint)
    with torch.device("meta"):
        model, _ = create_model(
            "satdino_b16", 4, pretrained=True, pretrained_checkpoint=checkpoint
        )
    assert model.encoder.gsd_register.shape == (1, 1, 768)


def test_satdino_rejects_checkpoint_missing_gsd_register(tmp_path):
    with torch.device("meta"):
        backbone = SatDINOEncoder()
    state = backbone.state_dict()
    del state["gsd_register"]
    checkpoint = tmp_path / "incomplete_teacher.pth"
    torch.save({"teacher": state}, checkpoint)
    with torch.device("meta"), pytest.raises(RuntimeError, match="gsd_register"):
        create_model("satdino_b16", 4, pretrained=True, pretrained_checkpoint=checkpoint)
