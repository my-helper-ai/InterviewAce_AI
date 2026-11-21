import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# FILE PATHS
# =========================================================

FUSION_FILE = Path(
    "data/fusion/InterviewAce_horizontal_fusion.csv"
)

OUTPUT_DIR = Path("data/quality")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(FUSION_FILE)

print("=" * 70)
print("DATA QUALITY ASSESSMENT")
print("=" * 70)

print("\nDataset loaded successfully.")


# =========================================================
# 1. DIMENSIONS
# =========================================================

print("\n" + "=" * 70)
print("1. DATASET DIMENSIONS")
print("=" * 70)

rows, columns = df.shape

print(f"Rows    : {rows}")
print(f"Columns : {columns}")


# =========================================================
# 2. DATA TYPES
# =========================================================

print("\n" + "=" * 70)
print("2. DATA TYPES")
print("=" * 70)

print(df.dtypes)


# =========================================================
# 3. UNIQUE VALUES
# =========================================================

print("\n" + "=" * 70)
print("3. UNIQUE VALUES")
print("=" * 70)

unique_summary = pd.DataFrame({
    "column": df.columns,
    "unique_values": [
        df[col].nunique(dropna=False)
        for col in df.columns
    ]
})

print(unique_summary.to_string(index=False))


# =========================================================
# 4. MISSING VALUES
# =========================================================

print("\n" + "=" * 70)
print("4. MISSING VALUES")
print("=" * 70)

missing_count = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
)

missing_summary = pd.DataFrame({
    "column": df.columns,
    "missing_count": missing_count.values,
    "missing_percentage": missing_percentage.values
})

print(missing_summary.to_string(index=False))


# =========================================================
# 5. DUPLICATE RECORDS
# =========================================================

print("\n" + "=" * 70)
print("5. DUPLICATE RECORDS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count}")


# =========================================================
# 6. DUPLICATE TECHNOLOGY KEYS
# =========================================================

print("\n" + "=" * 70)
print("6. DUPLICATE TECHNOLOGY KEYS")
print("=" * 70)

duplicate_keys = df["technology"].duplicated().sum()

print(f"Duplicate technology keys: {duplicate_keys}")


# =========================================================
# 7. DATA CONSISTENCY
# =========================================================

print("\n" + "=" * 70)
print("7. DATA CONSISTENCY CHECK")
print("=" * 70)

print("\nTechnology values:")

print(
    df["technology"]
    .value_counts()
    .to_string()
)


# Check negative values in numerical columns

numeric_columns = df.select_dtypes(
    include=np.number
).columns

print("\nNegative values:")

for col in numeric_columns:

    negative_count = (df[col] < 0).sum()

    print(
        f"{col}: {negative_count}"
    )


# =========================================================
# 8. OUTLIER DETECTION USING IQR
# =========================================================

print("\n" + "=" * 70)
print("8. OUTLIER DETECTION — IQR METHOD")
print("=" * 70)


outlier_results = []


for col in numeric_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ]

    outlier_count = len(outliers)

    outlier_results.append({
        "column": col,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_count
    })

    print(
        f"{col}: {outlier_count} outliers"
    )


outlier_df = pd.DataFrame(outlier_results)


# =========================================================
# 9. DESCRIPTIVE STATISTICS
# =========================================================

print("\n" + "=" * 70)
print("9. DESCRIPTIVE STATISTICS")
print("=" * 70)

statistics = df[numeric_columns].describe()

print(statistics)


# =========================================================
# 10. SAVE QUALITY REPORT
# =========================================================

print("\n" + "=" * 70)
print("SAVING QUALITY REPORT")
print("=" * 70)


excel_output = (
    OUTPUT_DIR /
    "InterviewAce_Data_Quality_Report.xlsx"
)


with pd.ExcelWriter(
    excel_output,
    engine="openpyxl"
) as writer:

    df.to_excel(
        writer,
        sheet_name="Fused_Data",
        index=False
    )

    unique_summary.to_excel(
        writer,
        sheet_name="Unique_Values",
        index=False
    )

    missing_summary.to_excel(
        writer,
        sheet_name="Missing_Values",
        index=False
    )

    outlier_df.to_excel(
        writer,
        sheet_name="Outliers_IQR",
        index=False
    )

    statistics.to_excel(
        writer,
        sheet_name="Statistics"
    )


print(f"\nQuality report saved to:")
print(excel_output)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("DATA QUALITY ASSESSMENT COMPLETED")
print("=" * 70)

print(f"""
Dataset dimensions : {rows} rows × {columns} columns
Duplicate rows     : {duplicate_count}
Duplicate keys     : {duplicate_keys}
Missing values     : {df.isnull().sum().sum()}
Numeric columns    : {len(numeric_columns)}
""")

print("=" * 70)