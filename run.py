import argparse
import sys

from src.mvp_pipeline import MVPPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps MVP runner")
    parser.add_argument("--mode", "-mode", required=True, choices=["inference", "update", "summary"])
    parser.add_argument("--file", "-file", default=None, help="Path to CSV for inference")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = MVPPipeline()

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
