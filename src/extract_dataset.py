"""
extract_dataset.py
------------------
1. Executes sql/create_heart_disease_dataset.sql against PostgreSQL
   (builds all intermediate tables + the final heart_disease_dataset table).
2. Loads heart_disease_dataset into a pandas DataFrame.
3. Prints a dataset summary (shape, columns, nulls, class distribution).
4. Saves the result to data/heart_disease_dataset.csv.

Run from the project root:
    python src/extract_dataset.py
"""

import os
import sys
import subprocess
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

# Allow running as  python src/extract_dataset.py  from project root
sys.path.insert(0, str(Path(__file__).parent))
from db_connection import get_engine  # noqa: E402

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT_DIR   = Path(__file__).resolve().parent.parent
SQL_FILE   = ROOT_DIR / "sql" / "create_heart_disease_dataset.sql"
DATA_DIR   = ROOT_DIR / "data"
OUTPUT_CSV = DATA_DIR / "heart_disease_dataset.csv"
DATA_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# Step 1 – Execute the SQL script via psql
# ------------------------------------------------------------------
def run_sql_script() -> None:
    """
    Run create_heart_disease_dataset.sql through psql.
    psql handles multi-statement scripts with inline comments correctly.
    Credentials are passed via environment variables — never on the command line.
    """
    if not SQL_FILE.exists():
        raise FileNotFoundError(f"SQL file not found: {SQL_FILE}")

    db_host     = os.getenv("DB_HOST", "localhost")
    db_port     = os.getenv("DB_PORT", "5432")
    db_name     = os.getenv("DB_NAME")
    db_user     = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")

    if not db_name or not db_user or not db_password:
        raise EnvironmentError(
            "DB_NAME, DB_USER, and DB_PASSWORD must be set in the .env file."
        )

    env = os.environ.copy()
    env["PGPASSWORD"] = db_password   # psql reads this env var; never passed as arg

    cmd = [
        "psql",
        "-h", db_host,
        "-p", db_port,
        "-U", db_user,
        "-d", db_name,
        "-f", str(SQL_FILE),
    ]

    print(f"Running SQL script: {SQL_FILE.name}")
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"psql exited with code {result.returncode}.\n"
            f"stderr:\n{result.stderr}"
        )

    # Print psql output so the user sees the CHECK results
    print(result.stdout)
    if result.stderr.strip():
        # psql NOTICE messages arrive on stderr — print but don't treat as error
        print(result.stderr)


# ------------------------------------------------------------------
# Step 2 – Load the final table into pandas
# ------------------------------------------------------------------
def load_dataset(engine) -> pd.DataFrame:
    print("Loading heart_disease_dataset from PostgreSQL...")
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                sql=text("SELECT * FROM heart_disease_dataset"),
                con=conn,
            )
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Failed to load heart_disease_dataset: {exc}") from exc
    return df


# ------------------------------------------------------------------
# Step 3 – Print summary
# ------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    sep = "-" * 60

    print(f"\n{sep}")
    print(f"Shape     : {df.shape[0]:,} rows × {df.shape[1]} columns")

    print(f"\n{sep}")
    print("Columns:")
    for col in df.columns:
        print(f"  {col}")

    print(f"\n{sep}")
    print("Missing values per column:")
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(1)
    null_report = pd.DataFrame({"nulls": null_counts, "pct": null_pct})
    null_report = null_report[null_report["nulls"] > 0].sort_values("nulls", ascending=False)
    if null_report.empty:
        print("  No missing values.")
    else:
        print(null_report.to_string())

    print(f"\n{sep}")
    print("heart_disease class distribution:")
    dist = (
        df["heart_disease"]
        .value_counts()
        .rename_axis("heart_disease")
        .reset_index(name="count")
    )
    dist["pct"] = (dist["count"] / len(df) * 100).round(2)
    print(dist.to_string(index=False))

    print(f"\n{sep}")
    print("hospital_expire_flag distribution:")
    mort = (
        df["hospital_expire_flag"]
        .value_counts()
        .rename_axis("hospital_expire_flag")
        .reset_index(name="count")
    )
    mort["pct"] = (mort["count"] / len(df) * 100).round(2)
    print(mort.to_string(index=False))
    print(sep)


# ------------------------------------------------------------------
# Step 4 – Save to CSV
# ------------------------------------------------------------------
def save_csv(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDataset saved → {OUTPUT_CSV}")
    print(f"  {df.shape[0]:,} rows, {df.shape[1]} columns, "
          f"{OUTPUT_CSV.stat().st_size / 1024:.1f} KB")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def run() -> pd.DataFrame:
    try:
        run_sql_script()
        engine = get_engine()
        df = load_dataset(engine)
        print_summary(df)
        save_csv(df)
        return df
    except (FileNotFoundError, EnvironmentError, RuntimeError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run()
