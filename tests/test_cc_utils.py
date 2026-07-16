"""model_training.cc_utils — pure clip-classification helpers (no model/torch forward pass)."""
import numpy as np
import pandas as pd

from model_training.cc_utils import (
    _get_clip_labels,
    _get_mask,
    _get_t_acc,
    _test_time_window,
)


def test_get_mask_marks_valid_positions():
    mask = _get_mask([2, 3], max_length=4)
    assert mask.tolist() == [[1, 1, 0, 0], [1, 1, 1, 0]]


def test_get_t_acc_per_time_position():
    y = [0, 1, 2, 0, 1, 2]
    y_hat = [0, 1, 9, 0, 9, 2]
    acc = _get_t_acc(y_hat, y, k_time=3)
    assert acc.tolist() == [1.0, 0.5, 0.5]


def test_get_clip_labels_testretest_is_zero_rest_incremental():
    timing = pd.DataFrame({
        "run": ["MOVIE1_7T_AP", "MOVIE1_7T_AP", "MOVIE2_7T_PA"],
        "clip_name": ["testretest1", "twomen", "bridgeville"],
    })
    assert _get_clip_labels(timing) == {"testretest1": 0, "twomen": 1, "bridgeville": 2}


def test_test_time_window_shifts_when_start_nonzero():
    df = pd.DataFrame({"timepoint": [0, 1, 2, 3, 4, 5], "v": [10, 11, 12, 13, 14, 15]})
    out = _test_time_window(df, range(2, 5))
    assert out["timepoint"].tolist() == [0, 1, 2]  # shifted by start=2
    assert out["v"].tolist() == [12, 13, 14]


def test_test_time_window_no_shift_when_start_zero():
    df = pd.DataFrame({"timepoint": [0, 1, 2, 3], "v": [10, 11, 12, 13]})
    out = _test_time_window(df, range(0, 3))
    assert out["timepoint"].tolist() == [0, 1, 2]
    assert out["v"].tolist() == [10, 11, 12]
