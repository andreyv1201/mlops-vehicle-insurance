import argparse
import os
import sys

from src.mvp_pipeline import Config, MVPPipeline


def _csv_tuple(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps MVP runner")
    parser.add_argument("--mode", "-mode", required=True, choices=["inference", "update", "summary"])
    parser.add_argument("--file", "-file", default=None, help="Path to CSV for inference")
    parser.add_argument("--data-files", default=os.getenv("MLOPS_DATA_FILES"), help="Comma-separated CSV files")
    parser.add_argument("--batch-size", type=int, default=_env_int("MLOPS_BATCH_SIZE", Config.batch_size))
    parser.add_argument(
        "--batches-per-update",
        type=int,
        default=_env_int("MLOPS_BATCHES_PER_UPDATE", Config.batches_per_update),
    )
    parser.add_argument(
        "--min-quality-non-null-ratio",
        type=float,
        default=_env_float("MLOPS_MIN_QUALITY_NON_NULL_RATIO", Config.min_quality_non_null_ratio),
    )
    parser.add_argument("--artifacts-dir", default=os.getenv("MLOPS_ARTIFACTS_DIR", Config.artifacts_dir))
    parser.add_argument("--raw-store-dir", default=os.getenv("MLOPS_RAW_STORE_DIR", Config.raw_store_dir))
    parser.add_argument("--models-dir", default=os.getenv("MLOPS_MODELS_DIR", Config.models_dir))
    parser.add_argument("--reports-dir", default=os.getenv("MLOPS_REPORTS_DIR", Config.reports_dir))
    parser.add_argument("--target-col", default=os.getenv("MLOPS_TARGET_COL", Config.target_col))
    parser.add_argument(
        "--model-candidates",
        default=os.getenv("MLOPS_MODEL_CANDIDATES", ",".join(Config.model_candidates)),
        help="Comma-separated model names for update mode",
    )
    parser.add_argument(
        "--random-forest-estimators",
        type=int,
        default=_env_int("MLOPS_RANDOM_FOREST_ESTIMATORS", Config.random_forest_estimators),
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    data_files = _csv_tuple(args.data_files) or Config.data_files
    model_candidates = _csv_tuple(args.model_candidates) or Config.model_candidates
    return Config(
        data_files=data_files,
        batch_size=args.batch_size,
        batches_per_update=args.batches_per_update,
        min_quality_non_null_ratio=args.min_quality_non_null_ratio,
        model_candidates=model_candidates,
        random_forest_estimators=args.random_forest_estimators,
        artifacts_dir=args.artifacts_dir,
        raw_store_dir=args.raw_store_dir,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        target_col=args.target_col,
    )


def main() -> int:
    args = parse_args()
    app = MVPPipeline(build_config(args))

    if args.mode == "update":
        ok = app.update()
        print(ok)
        return 0

    if args.mode == "inference":
        if not args.file:
            raise ValueError("Для режима inference нужен --file.")
        out = app.inference(args.file)
        print(out)
        return 0

    out = app.summary()
    print(out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
