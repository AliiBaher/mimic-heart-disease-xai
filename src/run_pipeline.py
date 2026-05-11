"""
run_pipeline.py
---------------
Orchestrates the full XAI heart-disease pipeline in order:

  Step 0: Test PostgreSQL connection
  Step 1: Extract dataset  (extract_dataset.py)
  Step 2: Train models     (train_models.py)
  Step 3: SHAP analysis    (shap_analysis.py)

Usage
-----
    python src/run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

# ── Project root is one level above this file ────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"


def _banner(text: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def _run_step(label: str, script: Path) -> None:
    """Run a Python script as a subprocess; abort the pipeline on failure."""
    _banner(label)
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\n[PIPELINE ABORTED] {label} failed with exit code {result.returncode}.",
              file=sys.stderr)
        sys.exit(result.returncode)
    print(f"\n[OK] {label} completed successfully.")


def step0_test_connection() -> None:
    """Import and call test_connection() directly (no subprocess needed)."""
    _banner("Step 0: Testing PostgreSQL connection...")

    # Add src/ to path so the import resolves correctly regardless of cwd
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    from db_connection import test_connection  # noqa: PLC0415

    if not test_connection():
        print("\n[PIPELINE ABORTED] Could not connect to PostgreSQL. "
              "Check that the database is running and .env credentials are correct.",
              file=sys.stderr)
        sys.exit(1)
    print("\n[OK] PostgreSQL connection successful.")


def main() -> None:
    print("\nStarting XAI Heart-Disease Pipeline")
    print(f"Project root : {ROOT}")
    print(f"Python       : {sys.executable}")

    step0_test_connection()
    _run_step("Step 1: Extracting dataset...",   SRC / "extract_dataset.py")
    _run_step("Step 2: Training models...",      SRC / "train_models.py")
    _run_step("Step 3: Running SHAP analysis...", SRC / "shap_analysis.py")

    _banner("Pipeline complete!")
    print("  All steps finished successfully.")
    print()


if __name__ == "__main__":
    main()
