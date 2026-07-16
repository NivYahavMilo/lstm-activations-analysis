"""statistical_analysis.math_functions — z_score and pearson_correlation."""
import numpy as np
import pandas as pd

from statistical_analysis.math_functions import pearson_correlation, z_score


def test_z_score_centers_and_scales_by_population_std():
    seq = np.array([1.0, 2.0, 3.0])
    z = z_score(seq)
    assert np.isclose(z.mean(), 0.0)
    # np.std defaults to population std (ddof=0)
    assert np.allclose(z, (seq - seq.mean()) / seq.std())


def test_z_score_constant_sequence_is_nan():
    # zero std -> division by zero -> nan (documents current behavior)
    z = z_score(np.array([5.0, 5.0, 5.0]))
    assert np.isnan(z).all()


def test_pearson_correlation_matches_pandas_corr():
    seq = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 4.0, 6.0, 8.0],    # perfectly correlated with a
        "c": [4.0, 3.0, 2.0, 1.0],    # perfectly anti-correlated with a
    })
    out = pearson_correlation(seq)
    # correct Pearson correlation matrix (cross-checked against pandas)
    pd.testing.assert_frame_equal(out, seq.corr())
    assert np.isclose(out.loc["a", "b"], 1.0)
    assert np.isclose(out.loc["a", "c"], -1.0)
    assert np.allclose(np.diag(out), 1.0)
