"""
train_models.py
---------------
Trains and evaluates three classifiers for heart-disease risk prediction.

Target : heart_disease  (1 = ICD-9 410-414 / 427 / 428 present, 0 = absent)

NOTE – hospital_expire_flag is kept as a feature here because it is a
       clinically recorded outcome that may correlate with severity, and
       some project goals use it as a supplementary risk signal.
       *** If your goal is to PREDICT mortality, drop it to avoid leakage. ***
       Uncomment the line in load_and_prepare() to exclude it.

Outputs
-------
  outputs/tables/model_performance.csv   — per-model metrics table
  outputs/figures/roc_<model>.png        — individual ROC curve per model
  outputs/figures/roc_all_models.png     — overlaid ROC curves
  outputs/figures/confusion_<model>.png  — confusion matrix per model
  outputs/best_model.joblib              — best model by ROC-AUC

Run from project root:
    python src/train_models.py
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_CSV    = ROOT_DIR / "data"  / "heart_disease_dataset.csv"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
TABLES_DIR  = ROOT_DIR / "outputs" / "tables"
BEST_MODEL  = ROOT_DIR / "outputs" / "best_model.joblib"

for d in (FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)

TARGET       = "heart_disease"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# ---------------------------------------------------------------------------
# 1. Load & prepare
# ---------------------------------------------------------------------------
def load_and_prepare() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_CSV}\n"
            "Run  python src/extract_dataset.py  first."
        )

    df = pd.read_csv(DATA_CSV)

    # Drop database identifier columns
    df.drop(columns=["subject_id", "hadm_id", "icustay_id"], errors="ignore", inplace=True)

    # NOTE: hospital_expire_flag is retained as a feature (clinical severity proxy).
    # To prevent leakage when predicting mortality, uncomment the next line:
    # df.drop(columns=["hospital_expire_flag"], errors="ignore", inplace=True)

    # Drop sub-label columns that are direct components of the target
    # (keeping them would cause leakage into heart_disease)
    df.drop(columns=["ischemic_hd", "dysrhythmia", "heart_failure"], errors="ignore", inplace=True)

    y = df.pop(TARGET).astype(int)
    X = df.copy()

    # Encode gender (M/F → 0/1)
    if "gender" in X.columns:
        le = LabelEncoder()
        X["gender"] = le.fit_transform(X["gender"].astype(str))

    print(f"Features        : {X.shape[1]}")
    print(f"Samples         : {X.shape[0]:,}")
    print(f"Class balance   : {dict(y.value_counts().sort_index())}")

    return X, y


# ---------------------------------------------------------------------------
# 2. Build preprocessing + model pipelines
# ---------------------------------------------------------------------------
def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Separate imputation + encoding strategies for numeric vs. categorical columns."""
    numeric_cols     = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    transformers = [
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), numeric_cols),
    ]
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
            ]),
            categorical_cols,
        ))

    return ColumnTransformer(transformers, remainder="drop")


def build_pipelines(X: pd.DataFrame) -> dict[str, Pipeline]:
    # Each model gets its own preprocessor instance via Pipeline
    return {
        "Logistic Regression": Pipeline([
            ("pre", _make_preprocessor(X)),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": Pipeline([
            ("pre", _make_preprocessor(X)),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("pre", _make_preprocessor(X)),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# 3. Evaluate a single model
# ---------------------------------------------------------------------------
def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"  ROC-AUC : {roc_auc:.4f}")

    return {
        "model":     name,
        "accuracy":  round(acc,       4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1_score":  round(f1,        4),
        "roc_auc":   round(roc_auc,   4),
        "_y_prob":   y_prob,           # kept for plotting, stripped before CSV
    }


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------
def _model_slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def plot_roc_single(name: str, y_test: pd.Series, y_prob: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = FIGURES_DIR / f"roc_{_model_slug(name)}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ROC curve saved       → {path}")


def plot_roc_all(results: list[dict], y_test: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")

    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["_y_prob"])
        ax.plot(fpr, tpr, lw=2, label=f"{r['model']}  (AUC={r['roc_auc']:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = FIGURES_DIR / "roc_all_models.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Combined ROC saved    → {path}")


def plot_confusion(name: str, y_test: pd.Series, y_pred: np.ndarray) -> None:
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    path = FIGURES_DIR / f"confusion_{_model_slug(name)}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {path}")


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------
def run() -> None:
    # ── Load data ──────────────────────────────────────────────────
    X, y = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"\nTrain : {len(X_train):,}  |  Test : {len(X_test):,}")

    # ── Train & evaluate ───────────────────────────────────────────
    pipelines = build_pipelines(X_train)
    results   = []

    for name, pipeline in pipelines.items():
        print(f"\nTraining  {name} ...")
        pipeline.fit(X_train, y_train)

        row = evaluate_model(name, pipeline, X_test, y_test)
        results.append({"pipeline": pipeline, **row})

        # Per-model plots
        y_pred = pipeline.predict(X_test)
        plot_roc_single(name, y_test, row["_y_prob"])
        plot_confusion(name, y_test, y_pred)

    # Overlaid ROC
    plot_roc_all(results, y_test)

    # ── Performance table ──────────────────────────────────────────
    perf_cols = ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
    perf_df   = pd.DataFrame([{k: r[k] for k in perf_cols} for r in results])

    perf_path = TABLES_DIR / "model_performance.csv"
    perf_df.to_csv(perf_path, index=False)
    print(f"\n{'─'*55}")
    print("Model performance summary:")
    print(perf_df.to_string(index=False))
    print(f"\n  Performance table saved → {perf_path}")

    # ── Save best model ────────────────────────────────────────────
    best = max(results, key=lambda r: r["roc_auc"])
    joblib.dump(best["pipeline"], BEST_MODEL)
    print(f"  Best model ({best['model']}, AUC={best['roc_auc']:.4f}) saved → {BEST_MODEL}")


if __name__ == "__main__":
    try:
        run()
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

