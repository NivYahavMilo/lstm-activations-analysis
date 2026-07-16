"""config.py path roots: repo-relative defaults + LSTM_DATA_DIR / LSTM_OUTPUT_DIR overrides."""
import importlib
import os

import pytest

import config as config_module

ENV_VARS = ["LSTM_DATA_DIR", "LSTM_OUTPUT_DIR"]


@pytest.fixture
def reload_config():
    """Reload config under the current environment; restore defaults afterwards."""
    def _reload():
        return importlib.reload(config_module)

    yield _reload
    for key in ENV_VARS:
        os.environ.pop(key, None)
    importlib.reload(config_module)


def test_defaults_are_repo_relative(reload_config, monkeypatch):
    for key in ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    c = reload_config()
    assert c.DATA_ROOT == c.ROOT_PATH
    assert c.OUTPUT_ROOT == c.ROOT_PATH
    assert c.FMRI_DATA == os.path.join(c.ROOT_PATH, "fmri_data")
    assert c.RESULTS_PATH == os.path.join(c.ROOT_PATH, "results")


def test_data_dir_override_moves_inputs_only(reload_config, monkeypatch):
    monkeypatch.delenv("LSTM_OUTPUT_DIR", raising=False)
    monkeypatch.setenv("LSTM_DATA_DIR", "/tmp/data")
    c = reload_config()
    assert c.FMRI_DATA == os.path.join("/tmp/data", "fmri_data")
    assert c.PARCEL_DIR == os.path.join("/tmp/data", "fmri_data", "cifti")
    assert c.MAPPINGS_PATH == os.path.join("/tmp/data", "mappings")
    # outputs stay under the repo root
    assert c.RESULTS_PATH == os.path.join(c.ROOT_PATH, "results")


def test_output_dir_override_moves_outputs_only(reload_config, monkeypatch):
    monkeypatch.delenv("LSTM_DATA_DIR", raising=False)
    monkeypatch.setenv("LSTM_OUTPUT_DIR", "/tmp/out")
    c = reload_config()
    assert c.RESULTS_PATH == os.path.join("/tmp/out", "results")
    assert c.MODELS_PATH == os.path.join("/tmp/out", "saved_models")
    assert c.CONNECTIVITY_FOLDER == os.path.join("/tmp/out", "fmri_connectivity_matrices")
    # inputs stay under the repo root
    assert c.FMRI_DATA == os.path.join(c.ROOT_PATH, "fmri_data")
