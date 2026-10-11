"""Command-line entry point.

Examples
--------
Real data (place 'Online Retail.xlsx' in data/raw/ first):
    python scripts/run_pipeline.py

Explicit file:
    python scripts/run_pipeline.py --input "path/to/Online Retail.xlsx"

Demo with SYNTHETIC data (not real findings):
    python scripts/run_pipeline.py --sample
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retainiq import config  # noqa: E402
from retainiq.data_loader import load_raw  # noqa: E402
from retainiq.pipeline import run_pipeline  # noqa: E402
from retainiq.synthetic import generate_synthetic_transactions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="RetainIQ pipeline")
    parser.add_argument("--input", help="Path to the raw .xlsx/.csv (default: search data/raw/)")
    parser.add_argument("--sample", action="store_true",
                        help="Use SYNTHETIC demo data instead of the real dataset")
    parser.add_argument("--output-dir", help="Where to write outputs "
                        "(default: outputs/real or outputs/sample)")
    parser.add_argument("--horizon-days", type=int, default=config.DEFAULT_HORIZON_DAYS)
    parser.add_argument("--n-iter", type=int, default=30, help="RandomizedSearchCV iterations")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    args = parser.parse_args()

    try:
        if args.sample:
            print("Generating SYNTHETIC demo data (not real findings)...")
            raw, label = generate_synthetic_transactions(), "synthetic"
        else:
            raw, label = load_raw(args.input), "real"
        out_dir = Path(args.output_dir) if args.output_dir else (
            config.OUTPUTS / ("sample" if args.sample else "real"))
        result = run_pipeline(raw, out_dir, horizon_days=args.horizon_days,
                              n_iter=args.n_iter, cv=args.cv, source_label=label)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result["cleaning_report"], indent=2))
    print(json.dumps(result["metrics"], indent=2))
    print(f"\nOutputs written to: {out_dir}")
    print("Launch the dashboard with:  streamlit run app/dashboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
