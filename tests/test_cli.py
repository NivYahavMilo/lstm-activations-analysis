"""CLI wiring: argument parsing and subcommand -> handler dispatch (no workflow executed)."""
import pytest

import cli


def _parse(argv):
    return cli.build_parser().parse_args(argv)


def test_requires_a_subcommand():
    with pytest.raises(SystemExit):
        _parse([])


def test_unknown_subcommand_errors():
    with pytest.raises(SystemExit):
        _parse(["nope"])


def test_train_defaults_and_overrides():
    a = _parse(["train"])
    assert a.handler is cli._run_train
    assert a.mode == "rest_between" and a.epochs is None and a.per_network is False
    b = _parse(["train", "--mode", "clips", "--epochs", "10", "--per-network"])
    assert b.mode == "clips" and b.epochs == 10 and b.per_network is True


def test_train_rejects_bad_mode():
    with pytest.raises(SystemExit):
        _parse(["train", "--mode", "banana"])


def test_infer_requires_modes_and_parses_range():
    with pytest.raises(SystemExit):
        _parse(["infer"])  # --model-mode / --inference required
    a = _parse(["infer", "--model-mode", "combined", "--inference", "rest_between",
                "--tr-range", "10", "20"])
    assert a.handler is cli._run_infer
    assert a.model_mode == "combined" and a.inference == "rest_between"
    assert a.tr_range == [10, 20]


def test_connectivity_and_relational_coding():
    assert _parse(["connectivity"]).handler is cli._run_connectivity
    rc = _parse(["relational-coding", "--shuffle", "--with-retest"])
    assert rc.handler is cli._run_relational_coding
    assert rc.shuffle is True and rc.with_retest is True
    assert _parse(["relational-coding"]).shuffle is False


def test_correlation_scope():
    assert _parse(["correlation"]).scope == "wb"
    a = _parse(["correlation", "--scope", "net"])
    assert a.handler is cli._run_correlation and a.scope == "net"
