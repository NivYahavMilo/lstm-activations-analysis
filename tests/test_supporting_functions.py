"""supporting_functions — pickle/csv I/O helpers (round-trip on a tmp path)."""
import pandas as pd

from supporting_functions import _dict_to_pkl, _load_csv, _load_pkl


def test_dict_to_pkl_load_pkl_roundtrip(tmp_path):
    data = {"a": 1, "b": [1, 2, 3]}
    stem = tmp_path / "obj"
    _dict_to_pkl(data, str(stem))            # appends .pkl
    loaded = _load_pkl(f"{stem}.pkl")
    assert loaded == data


def test_load_csv(tmp_path):
    csv = tmp_path / "t.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv, index=False)
    out = _load_csv(str(csv))
    assert out["a"].tolist() == [1, 2]
    assert out["b"].tolist() == [3, 4]
