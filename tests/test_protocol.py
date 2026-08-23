from crp_benchmark.models import paper_main_batch_size
from crp_benchmark.training import FSC_PROTOCOL, MAIN_PROTOCOL


def test_table_2_main_protocol():
    assert MAIN_PROTOCOL.max_epochs == 30
    assert MAIN_PROTOCOL.checkpoint_metric == "macro_f1"
    assert MAIN_PROTOCOL.early_stopping_patience == 5
    assert MAIN_PROTOCOL.min_delta == 1e-4


def test_table_2_fsc_protocol():
    assert FSC_PROTOCOL.max_epochs == 20
    assert FSC_PROTOCOL.checkpoint_metric == "selection_metric"
    assert FSC_PROTOCOL.early_stopping_patience == 3
    assert FSC_PROTOCOL.min_delta == 1e-4


def test_main_batch_sizes_are_in_published_range():
    assert paper_main_batch_size("swin_b") == 72
    assert paper_main_batch_size("effnet_b5") == 64
