"""Command-line interface for the Beijing PM2.5 pipeline."""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from . import config, data, eda, model, report
from .clean import clean, save_processed


def _banner(step: str) -> None:
    print(f"\n{'=' * 72}\n {step}\n{'=' * 72}")


def _prepare(verbose: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    raw = data.load_raw()
    validation = data.validate(raw)
    cleaned, cleaning_log = clean(raw)
    if verbose:
        print(f"[data] validation passed ({len(validation['checks'])} checks, "
              f"{validation['rows']:,} hourly records)")
        print(f"[clean] missing target {cleaning_log['target_missing']:,} "
              f"({cleaning_log['target_missing_share']:.2%}) in "
              f"{cleaning_log['missing_runs']['runs']} runs | "
              f"episodes >{config.EPISODE_THRESHOLD}: "
              f"{cleaning_log['episode_hours']:,} hours")
    return cleaned, cleaning_log, validation


def cmd_download(args: argparse.Namespace) -> int:
    _banner("STEP 1/3 - download & validate")
    path = data.download(force=args.force)
    print(f"  file: {path}")
    validation = data.validate(data.load_raw(path))
    for check in validation["checks"]:
        status = "PASS" if check["passed"] else ("FAIL" if check["critical"] else "warn")
        print(f"  [{status:>4}] {check['check']:<30} {check['detail']}")
    return 0


def cmd_eda(args: argparse.Namespace) -> int:
    _banner("STEP 2/3 - exploratory data analysis")
    cleaned, _, _ = _prepare()
    save_processed(cleaned)
    eda.run_eda(cleaned)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    _banner("STEP 3/3 - modelling, baselines and episode detection")
    cleaned, cleaning_log, validation = _prepare()
    save_processed(cleaned)
    summary = eda.compute_summary(cleaned)
    metrics = model.train(cleaned)
    report.write_reports(summary, metrics, cleaning_log, validation)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    cleaned, cleaning_log, validation = _prepare()
    save_processed(cleaned)

    _banner("STEP 2/3 - exploratory data analysis")
    summary = eda.run_eda(cleaned)

    _banner("STEP 3/3 - modelling, baselines and episode detection")
    metrics = model.train(cleaned)
    report.write_reports(summary, metrics, cleaning_log, validation)

    print(f"\n[done] end-to-end pipeline finished in {time.perf_counter() - started:.1f}s")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    bundle = model.load_model()
    frame = pd.read_csv(args.input, na_values=[config.MISSING_TOKEN])
    frame.columns = [column.strip() for column in frame.columns]
    if config.TIMESTAMP not in frame.columns:
        frame[config.TIMESTAMP] = pd.to_datetime(frame[config.TIME_COLUMNS])
    predictions = model.predict(frame, scenario=args.scenario, bundle=bundle)

    output = pd.DataFrame({
        "row_id": range(len(frame)),
        "predicted_pm25": predictions.round(1),
        "predicted_episode": (predictions > bundle["threshold"]).astype(int),
    })
    output.to_csv(args.output, index=False)
    print(f"[score] {len(output):,} rows -> {args.output} "
          f"({int(output['predicted_episode'].sum()):,} episode hours flagged)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airquality", description="Beijing PM2.5 forecasting pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download and validate")
    download.add_argument("--force", action="store_true", help="re-download")
    download.set_defaults(func=cmd_download)

    subparsers.add_parser("eda", help="run exploratory analysis").set_defaults(
        func=cmd_eda)
    subparsers.add_parser("train", help="train and evaluate").set_defaults(func=cmd_train)
    subparsers.add_parser("all", help="run the full pipeline").set_defaults(func=cmd_all)

    score = subparsers.add_parser("score", help="score a CSV with the saved model")
    score.add_argument("--input", required=True)
    score.add_argument("--output", default="predictions.csv")
    score.add_argument("--scenario", default="weather_only",
                       choices=list(config.SCENARIOS))
    score.set_defaults(func=cmd_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 - clean CLI failure
        print(f"\n[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
