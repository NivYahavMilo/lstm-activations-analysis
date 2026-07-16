"""statistical_analysis.matrices_ops.MatricesOperations — correlation / matrix helpers."""
import numpy as np
import pandas as pd
import pytest

from statistical_analysis.matrices_ops import MatricesOperations as MO


def test_is_symmetric():
    assert MO.is_symmetric(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert not MO.is_symmetric(np.array([[1.0, 2.0], [3.0, 1.0]]))


def test_correlation_matrix():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    corr = MO.correlation_matrix(df)
    assert np.isclose(corr.loc["a", "b"], -1.0)
    assert np.isclose(corr.loc["a", "a"], 1.0)


def test_cross_correlation_matrix_pairs_columns():
    m1 = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [4.0, 3, 2, 1]})
    m2 = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [1.0, 2, 3, 4]})
    out = MO.cross_correlation_matrix(m1, m2)
    assert np.isclose(out["a"], 1.0)   # identical column -> +1
    assert np.isclose(out["b"], -1.0)  # reversed column -> -1


def test_flatten_matrix():
    assert MO.flatten_matrix(np.array([[1, 2], [3, 4]])).tolist() == [1, 2, 3, 4]


def test_get_avg_matrix_over_iterable():
    mats = [np.array([[1.0, 1], [1, 1]]), np.array([[3.0, 3], [3, 3]])]
    assert np.allclose(MO.get_avg_matrix(iter(mats)), 2.0)


def test_get_avg_vector():
    assert np.isclose(MO.get_avg_vector(pd.Series([1.0, 2.0, 3.0])), 2.0)


def test_drop_symmetric_drops_diagonal_by_default():
    m = pd.DataFrame(np.array([[1.0, 2, 3], [2, 4, 5], [3, 5, 6]]))
    out = MO.drop_symmetric_side_of_a_matrix(m.copy(), drop_diagonal=True)
    # strictly-lower triangle (diagonal removed)
    assert out.tolist() == [2.0, 3.0, 5.0]


def test_drop_symmetric_keeps_diagonal_when_requested():
    m = pd.DataFrame(np.array([[1.0, 2, 3], [2, 4, 5], [3, 5, 6]]))
    out = MO.drop_symmetric_side_of_a_matrix(m.copy(), drop_diagonal=False)
    # full lower triangle including the diagonal
    assert out.tolist() == [1.0, 2.0, 4.0, 3.0, 5.0, 6.0]


def test_drop_symmetric_on_nonsymmetric_raises_value_error():
    m = pd.DataFrame(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError):
        MO.drop_symmetric_side_of_a_matrix(m)


def test_drop_symmetric_does_not_mutate_input():
    m = pd.DataFrame(np.array([[1.0, 2, 3], [2, 4, 5], [3, 5, 6]]))
    before = m.to_numpy().copy()
    MO.drop_symmetric_side_of_a_matrix(m, drop_diagonal=True)
    # the diagonal must be untouched (previous implementation wrote NaNs into it)
    assert np.array_equal(m.to_numpy(), before)
