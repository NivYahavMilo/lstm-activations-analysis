"""Command-line interface for the main LSTM-activations-analysis workflows.

Each subcommand fronts an existing entry point. Heavy modules (torch, the pipelines) are
imported lazily inside the handlers, so building the parser is cheap and side-effect free.

    python -m cli train --mode rest_between --epochs 50
    python -m cli infer --model-mode combined --inference rest_between
    python -m cli correlation --scope net
    python -m cli relational-coding --shuffle
    python -m cli connectivity

Workflows read/write data under the paths configured in settings.py (see .env.example),
so a real run needs the preprocessed fMRI tables / trained models on disk. Raw-data
parcellation has its own script: ``python parcellation/parcellation.py --help``.
"""
import argparse

MODES = ["clips", "rest_between", "combined"]


def _run_train(args):
    from enums import Mode
    from model_training import train_lstm
    from model_training.hyperparameters import HyperParams

    hp = HyperParams()
    hp.mode = Mode(args.mode)
    if args.epochs is not None:
        hp.num_epochs = args.epochs
    if args.batch_size is not None:
        hp.batch_size = args.batch_size
    (train_lstm.run_net if args.per_network else train_lstm.run)(hp)


def _run_infer(args):
    from enums import Mode
    from model_training import inference

    tr_range = tuple(args.tr_range) if args.tr_range else ()
    results, res_path = inference._test(
        model_mode=Mode(args.model_mode), inference=Mode(args.inference), tr_range=tr_range)
    inference._save(results, res_path)


def _run_connectivity(args):
    from fmri_connectivity_matrices.networks_connectivity import Connectivity

    Connectivity.generate_connectivity_matrices()


def _run_relational_coding(args):
    from relational_coding.relational_coding_fmri import RelationalCodingfMRI

    RelationalCodingfMRI.shuffle = args.shuffle
    RelationalCodingfMRI.executor(with_retest=args.with_retest)


def _run_correlation(args):
    from statistical_analysis import correlation_pipelines as cp

    (cp.wb_pipeline if args.scope == "wb" else cp.net_pipeline)()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="lstm-activations-analysis",
        description="Train the LSTM and run the activation / relational-coding analyses.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train the clip-classification LSTM")
    p.add_argument("--mode", choices=MODES, default="rest_between", help="data mode to train on")
    p.add_argument("--epochs", type=int, default=None, help="override HyperParams.num_epochs")
    p.add_argument("--batch-size", type=int, default=None, help="override HyperParams.batch_size")
    p.add_argument("--per-network", action="store_true",
                   help="train one model per network (run_net) instead of the 300-ROI model")
    p.set_defaults(handler=_run_train)

    p = sub.add_parser("infer", help="run a trained model and save test results / activations")
    p.add_argument("--model-mode", choices=MODES, required=True, help="which trained model to load")
    p.add_argument("--inference", choices=MODES, required=True, help="data mode to run inference on")
    p.add_argument("--tr-range", type=int, nargs=2, metavar=("START", "END"), default=None,
                   help="optional TR window, e.g. --tr-range 10 20")
    p.set_defaults(handler=_run_infer)

    p = sub.add_parser("connectivity", help="generate network connectivity matrices")
    p.set_defaults(handler=_run_connectivity)

    p = sub.add_parser("relational-coding", help="fMRI relational-coding distances")
    p.add_argument("--shuffle", action="store_true", help="shuffle clip labels (permutation control)")
    p.add_argument("--with-retest", action="store_true", help="include the test-retest clips")
    p.set_defaults(handler=_run_relational_coding)

    p = sub.add_parser("correlation", help="activation correlation pipelines")
    p.add_argument("--scope", choices=["wb", "net"], default="wb",
                   help="whole-brain (wb) or per-network (net) pipeline")
    p.set_defaults(handler=_run_correlation)

    return parser


def run(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    run()
