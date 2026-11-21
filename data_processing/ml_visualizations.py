from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "ml_results"
OUTPUT_DIR = ROOT / "data" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Regression model comparison
regression_file = RESULTS_DIR / "model_evaluation.csv"
regression = pd.read_csv(regression_file)

print("\nRegression model evaluation:")
print(regression.to_string(index=False))

# MAE comparison
mae_column = next(
    (col for col in regression.columns if col.lower() == "mae"),
    None
)

if mae_column:
    plt.figure(figsize=(10, 6))
    plt.bar(regression.iloc[:, 0].astype(str), regression[mae_column])
    plt.title("Regression Models: Mean Absolute Error")
    plt.xlabel("Model")
    plt.ylabel("MAE (lower is better)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ml_mae_comparison.png", dpi=300)
    plt.close()

# R-squared comparison
r2_column = next(
    (col for col in regression.columns if "r2" in col.lower() or "r²" in col.lower()),
    None
)

if r2_column:
    plt.figure(figsize=(10, 6))
    plt.bar(regression.iloc[:, 0].astype(str), regression[r2_column])
    plt.axhline(0, linewidth=1)
    plt.title("Regression Models: R-squared")
    plt.xlabel("Model")
    plt.ylabel("R-squared (higher is better)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ml_r2_comparison.png", dpi=300)
    plt.close()

# 2. Classification model metrics
classification_file = RESULTS_DIR / "classification_evaluation.csv"
classification = pd.read_csv(classification_file)

print("\nClassification evaluation:")
print(classification.to_string(index=False))

metric_columns = [
    col for col in classification.columns
    if col.lower() in {"accuracy", "precision", "recall", "f1", "f1_score", "f1-score"}
]

if metric_columns:
    row = classification.iloc[0]
    labels = [col.replace("_", " ").title() for col in metric_columns]
    values = [row[col] for col in metric_columns]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.title("Classification Model Performance")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ml_classification_metrics.png", dpi=300)
    plt.close()

print("\nVisualization step completed.")
print(f"Charts saved in: {OUTPUT_DIR}")