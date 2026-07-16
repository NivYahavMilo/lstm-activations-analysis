"""relational_coding.relational_coding_base.RelationalCoding — pure helpers."""
import numpy as np
import pandas as pd

from relational_coding.relational_coding_base import RelationalCoding as RC


def _clip_table():
    return pd.DataFrame({
        "Subject": ["s1"] * 6,
        "y": [1, 1, 1, 2, 2, 2],
        "timepoint": [0, 1, 2, 0, 1, 2],
        "v0": [10, 11, 12, 20, 21, 22],
        "v1": [30, 31, 32, 40, 41, 42],
    })


def test_get_single_tr_defaults_to_last_timepoint():
    out = RC.get_single_tr(_clip_table(), clip_i=1, tr_field="timepoint")
    assert list(out.columns) == ["v0", "v1"]     # metadata columns dropped
    assert out.values.tolist() == [[12, 32]]      # last TR (timepoint 2) of clip 1


def test_get_single_tr_explicit_timepoint():
    out = RC.get_single_tr(_clip_table(), clip_i=2, tr_field="timepoint", tr_pos=0)
    assert out.values.tolist() == [[20, 40]]


def test_relational_distance_of_correlated_quadrants():
    # 6x6 with symmetric clip quadrant (top-left) and rest quadrant (bottom-right) whose
    # strictly-lower triangles are perfectly correlated ([1,2,3] vs [2,4,6]) -> distance 1.0
    clip = np.array([[1, 1, 2], [1, 1, 3], [2, 3, 1]], float)
    rest = np.array([[1, 2, 4], [2, 1, 6], [4, 6, 1]], float)
    full = np.zeros((6, 6))
    full[:3, :3] = clip
    full[3:, 3:] = rest
    assert RC.relational_distance(pd.DataFrame(full)) == 1.0


def test_shuffle_clips_mapping_wraps_around():
    assert RC._shuffle_clips(1) == 3
    assert RC._shuffle_clips(13) == 1
    assert RC._shuffle_clips(14) == 2
    assert RC._shuffle_clips(99) is None  # unknown index -> None
