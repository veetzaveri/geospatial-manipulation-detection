import numpy as np
from PIL import Image

from crp_benchmark.generation import create_local_mask, feathered_hard_composite


def test_mask_is_deterministic_and_local():
    first = np.asarray(create_local_mask("sample.png", seed=42))
    second = np.asarray(create_local_mask("sample.png", seed=42))
    assert np.array_equal(first, second)
    assert 0.0 < first.mean() / 255.0 <= 0.35


def test_feathered_composite_preserves_far_outside_pixels():
    original = Image.new("RGB", (64, 64), (10, 20, 30))
    generated = Image.new("RGB", (64, 64), (200, 210, 220))
    mask = Image.new("L", (64, 64), 0)
    pixels = np.asarray(mask).copy()
    pixels[24:40, 24:40] = 255
    mask = Image.fromarray(pixels)
    result = np.asarray(feathered_hard_composite(original, generated, mask, feather_radius=2))
    assert np.array_equal(result[0, 0], np.array([10, 20, 30]))
    assert result[32, 32, 0] > 150
