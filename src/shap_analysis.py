"""
shap_analysis.py
----------------
Applies SHAP explainability to the best trained model saved by train_models.py.

What are SHAP values?
---------------------
SHAP (SHapley Additive exPlanations) assigns each feature a contribution value
for every individual prediction.  A positive SHAP value pushes the prediction
towards the positive class (heart disease = 1); a negative value pushes it
towards 0.  The magnitude shows how strongly that feature influenced the
prediction for that specific patient.

Outputs
-------
  outputs/figures/shap_summary.png            — beeswarm: feature impact & direction
  outputs/figures/shap_feature_importance.png — bar chart: mean |SHAP| per feature
  outputs/tables/shap_top15_features.csv      — ranked top-15 features by mean |SHAP|

Run from project root:
    python src/shap_analysis.py
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_CSV    = ROOT_DIR / "data"   / "heart_disease_dataset.csv"
MODEL_FILE  = ROOT_DIR / "outputs" / "best_model.joblib"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
TABLES_DIR  = ROOT_DIR / "outputs" / "tables"

for d in (FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)

TARGET      = "heart_disease"
DROP_COLS   = ["subject_id", "hadm_id", "icustay_id",
               "ischemic_hd", "dysrhythmia", "heart_failure"]
SAMPLE_SIZE = 1000   # rows used for SHAP (caps runtime; increase for more detail)


# ---------------------------------------------------------------------------
# 1. Load data — mirror the exact feature engineering in train_models.py
# ---------------------------------------------------------------------------
def load_features() -> pd.DataFrame:
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_CSV}\n"
            "Run  python src/extract_dataset.py  first."
        )

    df = pd.read_csv(DATA_CSV)
    df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

    # Remove the target — we only need X here
    df.drop(columns=[TARGET], errors="ignore", inplace=True)

    # Mirror train_models.py: encode gender with LabelEncoder (M/F → 0/1)
    # before the data enters the sklearn Pipeline, which expects gender numeric.
    if "gender" in df.columns:
        le = LabelEncoder()
        df["gender"] = le.fit_transform(df["gender"].astype(str))

    return df


# ---------------------------------------------------------------------------
# 2. Load the saved sklearn Pipeline
# ---------------------------------------------------------------------------
def load_pipeline():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}\n"
            "Run  python src/train_models.py  first."
        )
    pipeline = joblib.load(MODEL_FILE)
    print(f"Loaded model : {MODEL_FILE.name}")
    clf = pipeline.named_steps["clf"]
    print(f"Classifier   : {type(clf).__name__}")
    return pipeline, clf


# ---------------------------------------------------------------------------
# 3. Transform X through the preprocessing steps (all steps except final clf)
# ---------------------------------------------------------------------------
def transform_features(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply every pipeline step except the classifier."""
    Xt = X
    for name, step in pipeline.steps[:-1]:
        Xt = step.transform(Xt)
    return Xt


def get_feature_names(pipeline, X: pd.DataFrame) -> list[str]:
    """
    Recover output feature names from the ColumnTransformer ('pre' step).
    Falls back to positional names if get_feature_names_out() is unavailable.
    """
    pre = pipeline.named_steps["pre"]
    try:
        names = list(pre.get_feature_names_out())
        # Strip transformer prefixes added by sklearn (e.g. 'num__age' → 'age')
        names = [n.split("__", 1)[-1] for n in names]
    except AttributeError:
        # Older sklearn: build names manually from ColumnTransformer
        names = []
        for tname, _, cols in pre.transformers_:
            if tname == "remainder":
                continue
            names.extend(cols if hasattr(cols, "__iter__") and not isinstance(cols, str) else [cols])
    return names


# ---------------------------------------------------------------------------
# 4. Build the right SHAP explainer for the classifier type
# ---------------------------------------------------------------------------
def build_explainer(clf, X_transformed: np.ndarray, feature_names: list[str]):
    """
    Select the most appropriate SHAP explainer:
      - TreeExplainer  : fast exact Shapley values for tree-based models
                         (RandomForest, GradientBoosting, XGBoost, LightGBM …)
      - LinearExplainer: exact values for linear models (LogisticRegression)
      - Explainer      : model-agnostic fallback (slower)

    Returns (explainer, shap_values_2d) where shap_values_2d has shape
    (n_samples, n_features) for the positive class.
    """
    clf_type = type(clf).__name__
    print(f"Building explainer for {clf_type} ...")

    tree_types   = {"RandomForestClassifier", "GradientBoostingClassifier",
                    "ExtraTreesClassifier",   "XGBClassifier",
                    "LGBMClassifier",         "CatBoostClassifier",
                    "DecisionTreeClassifier"}
    linear_types = {"LogisticRegression", "LinearSVC", "SGDClassifier",
                    "RidgeClassifier"}

    bg = shap.sample(X_transformed, min(200, len(X_transformed)))   # background

    if clf_type in tree_types:
        explainer   = shap.TreeExplainer(clf)
        shap_out    = explainer.shap_values(X_transformed)
        # TreeExplainer returns list [class0, class1] for binary classifiers
        shap_values = shap_out[1] if isinstance(shap_out, list) else shap_out

    elif clf_type in linear_types:
        explainer   = shap.LinearExplainer(clf, bg)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    else:
        # Generic model-agnostic explainer (slower — uses sampling)
        print("  Using model-agnostic Explainer (this may take a moment) ...")
        explainer   = shap.Explainer(clf.predict_proba, bg)
        shap_out    = explainer(X_transformed)
        # shap_out.values has shape (n, features, classes) for multi-output
        shap_values = (shap_out.values[:, :, 1]
                       if shap_out.values.ndim == 3
                       else shap_out.values)

    print(f"  SHAP values shape : {shap_values.shape}")
    return explainer, shap_values


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
def plot_summary_beeswarm(shap_values: np.ndarray,
                          X_transformed: np.ndarray,
                          feature_names: list[str]) -> None:
    """
    Beeswarm summary plot.

    Each dot represents one patient. The x-axis shows the SHAP value
    (impact on the model's output). The colour shows the original feature
    value (red = high, blue = low).  Features are sorted by mean |SHAP|,
    so the most influential features appear at the top.
    """
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    fig, _ = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_df,
        plot_type="dot",
        show=False,
        max_display=20,
    )
    plt.title("SHAP Summary — Feature Impact on Heart Disease Prediction",
              fontsize=12, pad=12)
    plt.tight_layout()
    path = FIGURES_DIR / "shap_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary plot saved → {path}")


def plot_bar_importance(shap_values: np.ndarray,
                        feature_names: list[str]) -> pd.DataFrame:
    """
    Bar chart of mean absolute SHAP values.

    The bar length represents the average magnitude of a feature's
    contribution across all patients — a global importance ranking.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    top = importance.head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|  (average impact on model output)")
    ax.set_title("SHAP Feature Importance — Top 15 Features")
    plt.tight_layout()
    path = FIGURES_DIR / "shap_feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart saved    → {path}")

    return importance


# ---------------------------------------------------------------------------
# 6. Save importance table
# ---------------------------------------------------------------------------
def save_top15_csv(importance: pd.DataFrame) -> None:
    top15 = importance.head(15).copy()
    top15.index = range(1, len(top15) + 1)
    top15.index.name = "rank"
    path = TABLES_DIR / "shap_top15_features.csv"
    top15.to_csv(path)
    print(f"  Top-15 table saved → {path}")

    print("\nTop 15 features by mean |SHAP value|:")
    print(top15.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run() -> None:
    # Load
    X = load_features()
    pipeline, clf = load_pipeline()

    # Sample for SHAP (full dataset can be slow for tree models too)
    n = min(SAMPLE_SIZE, len(X))
    X_sample = X.sample(n=n, random_state=42).reset_index(drop=True)
    print(f"\nUsing {n:,} samples for SHAP analysis (of {len(X):,} total)")

    # Preprocess
    X_transformed = transform_features(pipeline, X_sample)
    feature_names = get_feature_names(pipeline, X_sample)
    print(f"Transformed shape : {X_transformed.shape}")
    print(f"Feature names     : {feature_names}")

    # SHAP
    _, shap_values = build_explainer(clf, X_transformed, feature_names)

    # Plots & table
    print()
    plot_summary_beeswarm(shap_values, X_transformed, feature_names)
    importance = plot_bar_importance(shap_values, feature_names)
    save_top15_csv(importance)

    print("\nSHAP analysis complete.")


if __name__ == "__main__":
    try:
        run()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

