from pathlib import Path
import html

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "api_data" / "stackexchange_python.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "ml_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Clean text
df["title"] = df["title"].fillna("").map(html.unescape)
df["tags"] = df["tags"].fillna("").str.replace(",", " ", regex=False)
df["answer_count"] = pd.to_numeric(df["answer_count"], errors="coerce")
df["score"] = pd.to_numeric(df["score"], errors="coerce")

# Keep usable records
df = df.dropna(subset=["score", "answer_count"]).copy()

# Combine title and tags for text features
df["text_features"] = df["title"] + " " + df["tags"]

X = df[["answer_count", "text_features"]]
y = df["score"]

# Split before model fitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Process numeric and text features
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), ["answer_count"]),
        ("text", TfidfVectorizer(max_features=1000, ngram_range=(1, 2)),
         "text_features")
    ]
)

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=3,
        n_jobs=-1
    ),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        min_samples_leaf=5
    ),
    
        "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    )
}

results = []

for model_name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"\n{model_name}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

# Save evaluation results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "model_evaluation.csv", index=False)

print("\nML experiment completed.")
print(f"Results saved to: {OUTPUT_DIR / 'model_evaluation.csv'}")

# ==========================================
# RANDOM FOREST CLASSIFICATION EXPERIMENT
# ==========================================

print("\n" + "=" * 40)
print("RANDOM FOREST CLASSIFICATION")
print("=" * 40)

from sklearn.metrics import classification_report, confusion_matrix

# Create three score categories
df["score_category"] = pd.qcut(
    df["score"],
    q=3,
    labels=["Low", "Medium", "High"]
)

# Features and target
X_class = X.copy()
y_class = df["score_category"]

# Split the data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class,
    y_class,
    test_size=0.20,
    random_state=42,
    stratify=y_class
)

# Build the classification pipeline
classifier_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1
    ))
])

# Train and predict
classifier_pipeline.fit(X_train_c, y_train_c)
class_predictions = classifier_pipeline.predict(X_test_c)

# Evaluate
accuracy = accuracy_score(y_test_c, class_predictions)
precision = precision_score(
    y_test_c, class_predictions, average="weighted", zero_division=0
)
recall = recall_score(
    y_test_c, class_predictions, average="weighted", zero_division=0
)
f1 = f1_score(
    y_test_c, class_predictions, average="weighted", zero_division=0
)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")

print("\nClassification Report:")
print(classification_report(
    y_test_c, class_predictions, zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(
    y_test_c, class_predictions, labels=["Low", "Medium", "High"]
))

# Save classification results separately
classification_results = pd.DataFrame([{
    "model": "Random Forest Classifier",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1
}])

classification_results.to_csv(
    OUTPUT_DIR / "classification_evaluation.csv",
    index=False
)

print("\nClassification results saved to:")
print(OUTPUT_DIR / "classification_evaluation.csv")