"""CLI entry point."""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="barbell", description="Barbell tracker CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-data", help="Ingest raw CSVs into a resampled pickle")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default=None)

    p = sub.add_parser("train", help="Train models with subject-held-out CV")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default="hist_gradient_boosting")
    p.add_argument("--out", default="models/model.joblib")

    p = sub.add_parser("eval-reps", help="Evaluate repetition counting")
    p.add_argument("--config", default="configs/default.yaml")

    args = parser.parse_args(argv)
    if args.cmd == "build-data":
        from barbell.data.ingestion import build_dataset, validate_dataset
        from barbell.utils.config import load_config, project_root
        cfg = load_config(args.config)
        root = project_root(cfg)
        raw = root / cfg["paths"]["raw_dir"]
        out = Path(args.out) if args.out else root / cfg["paths"]["interim_dir"] / "01_data_processed.pkl"
        df = build_dataset(raw, resample_ms=cfg["data"]["resample_ms"])
        validate_dataset(df)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(out)
        print(f"Wrote {out} {df.shape}")
    elif args.cmd == "train":
        print("Use scripts/train.py for the full experiment; CLI train is a thin wrapper.")
        return 1
    elif args.cmd == "eval-reps":
        print("Use scripts/eval_reps.py")
        return 1
    return 0


if __name__ == "__main__":
    from pathlib import Path
    sys.exit(main())