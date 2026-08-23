from crp_benchmark.models import MODEL_SPECS, resolve_model_key


def test_registry_contains_paper_20_models():
    assert len(MODEL_SPECS) == 20


def test_aliases():
    assert resolve_model_key("Swin-B") == "swin_b"
    assert resolve_model_key("GeoRSCLIP") == "georsclip_b32"
