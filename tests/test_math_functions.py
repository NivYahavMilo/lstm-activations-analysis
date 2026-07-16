"""statistical_analysis.math_functions — z_score and pearson_correlation.

These pin *current* behavior. `pearson_correlation` is almost certainly buggy (see the note on
its test) and is revisited in the bug-fix phase.
"""
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


def test_pearson_correlation_current_behavior():
    # BUG (pinned): `(1/len(seq)-1)` parses as (1/len)-1 rather than 1/(len-1), and
    # `seq.T * seq` is an element-wise product, not a matrix/correlation product. So this
    # does not compute a Pearson correlation. Locked here as-is; fixed in the bug-fix phase.
    seq = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
    out = pearson_correlation(seq)
    expected = (1 / len(seq) - 1) * (seq.T * seq)
    pd.testing.assert_frame_equal(out, expected)
    # concretely, for this input:
    assert out.values.tolist() == [[-0.5, -3.0], [-3.0, -8.0]]
