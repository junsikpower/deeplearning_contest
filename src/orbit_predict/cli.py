"""Command-line entry point for data checks and the full experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import DatasetPaths, load_datasets
from .experiment import run_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orbit satellite-position prediction pipeline")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("orbit-predict-satellite-position/Train_DS..csv"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("orbit-predict-satellite-position/Test_DS..csv"),
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("orbit-predict-satellite-position/Sample_Submission_DS..csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--report", type=Path, default=Path("docs/model-report.md"))
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="validate the three supplied CSVs without fitting models",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="run both models, selection, validation, and submission generation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = DatasetPaths(
        train=args.train,
        test=args.test,
        sample_submission=args.sample_submission,
    )
    if args.check_inputs == args.run:
        raise SystemExit("choose exactly one of --check-inputs or --run")
    if args.check_inputs:
        datasets = load_datasets(paths)
        print(
            f"input contract OK: train={len(datasets.train):,}, "
            f"test={len(datasets.test):,}, sample={len(datasets.sample_submission):,}"
        )
        return 0
    result = run_experiment(
        paths,
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        report_path=args.report,
    )
    print(f"experiment complete: {len(result.metrics)} comparison stages")
    for output in result.outputs.values():
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

