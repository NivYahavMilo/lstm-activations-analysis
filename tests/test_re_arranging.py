"""mappings.re_arranging.rearrange_clips — clip/rest column (or row) ordering."""
import pandas as pd

from mappings.re_arranging import rearrange_clips

CLIPS = ["twomen", "bridgeville", "pockets", "overcome", "inception", "socialnet", "oceans",
         "flower", "hotel", "garden", "dreary", "homealone", "brokovich", "starwars"]


def _expected_order(with_testretest):
    clips = [f"{c}_clips" for c in CLIPS]
    rests = [f"{c}_rest_between" for c in CLIPS]
    if with_testretest:
        clips += [f"testretest{i}_clips" for i in (1, 2, 3, 4)]
        rests += [f"testretest{i}_rest_between" for i in (1, 2, 3, 4)]
    return clips + rests


def test_rearrange_columns_default_no_testretest():
    exp = _expected_order(with_testretest=False)
    df = pd.DataFrame([[0] * len(exp)], columns=list(reversed(exp)))
    out = rearrange_clips(df)
    assert list(out.columns) == exp


def test_rearrange_columns_with_testretest():
    exp = _expected_order(with_testretest=True)
    df = pd.DataFrame([[0] * len(exp)], columns=list(reversed(exp)))
    out = rearrange_clips(df, with_testretest=True)
    assert list(out.columns) == exp


def test_rearrange_rows():
    exp = _expected_order(with_testretest=False)
    df = pd.DataFrame({"v": [0] * len(exp)}, index=list(reversed(exp)))
    out = rearrange_clips(df, where="rows")
    assert list(out.index) == exp
