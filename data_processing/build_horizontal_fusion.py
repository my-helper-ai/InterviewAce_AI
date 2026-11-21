import pandas as pd
from pathlib import Path


# =========================================================
# FOLDERS
# =========================================================

CLEAN_DIR = Path("data/cleaned")
FUSION_DIR = Path("data/fusion")

FUSION_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPER FUNCTION
# =========================================================

def load_files(pattern):
    files = list(CLEAN_DIR.glob(pattern))

    dataframes = []

    for file in files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            print(f"Loaded: {file.name} ({len(df)} rows)")
        except Exception as e:
            print(f"Could not load {file.name}: {e}")

    if dataframes:
        return pd.concat(dataframes, ignore_index=True)

    return pd.DataFrame()


# =========================================================
# STACK EXCHANGE
# =========================================================

print("\n" + "=" * 70)
print("STACK EXCHANGE DATA")
print("=" * 70)

stack_df = load_files("stackexchange_*.csv")

print(f"\nTotal Stack Exchange records: {len(stack_df)}")


# Ensure numeric columns
stack_df["score"] = pd.to_numeric(
    stack_df["score"], errors="coerce"
).fillna(0)

stack_df["answer_count"] = pd.to_numeric(
    stack_df["answer_count"], errors="coerce"
).fillna(0)


# Aggregate Stack Exchange by technology
stack_fused = (
    stack_df
    .groupby("technology")
    .agg(
        stack_question_count=("question_id", "count"),
        stack_avg_score=("score", "mean"),
        stack_max_score=("score", "max"),
        stack_avg_answers=("answer_count", "mean"),
        stack_total_answers=("answer_count", "sum")
    )
    .reset_index()
)


# =========================================================
# GITHUB
# =========================================================

print("\n" + "=" * 70)
print("GITHUB DATA")
print("=" * 70)

github_df = load_files("github_*.csv")

print(f"\nTotal GitHub records: {len(github_df)}")


github_df["stars"] = pd.to_numeric(
    github_df["stars"], errors="coerce"
).fillna(0)

github_df["forks"] = pd.to_numeric(
    github_df["forks"], errors="coerce"
).fillna(0)


github_fused = (
    github_df
    .groupby("technology")
    .agg(
        github_repository_count=("repository", "count"),
        github_avg_stars=("stars", "mean"),
        github_total_stars=("stars", "sum"),
        github_avg_forks=("forks", "mean"),
        github_total_forks=("forks", "sum")
    )
    .reset_index()
)


# =========================================================
# REMOTIVE
# =========================================================

print("\n" + "=" * 70)
print("REMOTIVE DATA")
print("=" * 70)

remotive_df = load_files("remotive_*.csv")

print(f"\nTotal Remotive records: {len(remotive_df)}")


# Count jobs
remotive_fused = (
    remotive_df
    .groupby("technology")
    .agg(
        remotive_job_count=("job_id", "count"),
        remotive_company_count=("company", "nunique"),
        remotive_location_count=("location", "nunique")
    )
    .reset_index()
)


# =========================================================
# HORIZONTAL FUSION
# =========================================================

print("\n" + "=" * 70)
print("PERFORMING HORIZONTAL DATA FUSION")
print("=" * 70)


# Stack Exchange + GitHub
fused_df = pd.merge(
    stack_fused,
    github_fused,
    on="technology",
    how="outer"
)


# Add Remotive
fused_df = pd.merge(
    fused_df,
    remotive_fused,
    on="technology",
    how="outer"
)


# =========================================================
# MISSING VALUES
# =========================================================

numeric_columns = fused_df.select_dtypes(
    include=["number"]
).columns

fused_df[numeric_columns] = fused_df[numeric_columns].fillna(0)


# =========================================================
# DERIVED FEATURES
# =========================================================

fused_df["total_activity"] = (
    fused_df["stack_question_count"]
    + fused_df["github_repository_count"]
    + fused_df["remotive_job_count"]
)


fused_df["engagement_score"] = (
    fused_df["stack_avg_score"]
    + fused_df["github_avg_stars"]
)


# =========================================================
# SORT
# =========================================================

fused_df = fused_df.sort_values(
    by="total_activity",
    ascending=False
).reset_index(drop=True)


# =========================================================
# SAVE CSV
# =========================================================

csv_output = FUSION_DIR / "InterviewAce_horizontal_fusion.csv"

fused_df.to_csv(
    csv_output,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# SAVE EXCEL
# =========================================================

excel_output = FUSION_DIR / "InterviewAce_API_Data_Fusion.xlsx"

with pd.ExcelWriter(
    excel_output,
    engine="openpyxl"
) as writer:

    fused_df.to_excel(
        writer,
        sheet_name="Fused_Data",
        index=False
    )

    stack_fused.to_excel(
        writer,
        sheet_name="StackExchange_Summary",
        index=False
    )

    github_fused.to_excel(
        writer,
        sheet_name="GitHub_Summary",
        index=False
    )

    remotive_fused.to_excel(
        writer,
        sheet_name="Remotive_Summary",
        index=False
    )


# =========================================================
# DISPLAY RESULT
# =========================================================

print("\n" + "=" * 70)
print("FUSION COMPLETED")
print("=" * 70)

print(f"\nFinal rows: {len(fused_df)}")
print(f"Final columns: {len(fused_df.columns)}")

print("\nFinal columns:")
print(list(fused_df.columns))

print("\nFused dataset:")
print(fused_df.to_string(index=False))

print("\nSaved CSV:")
print(csv_output)

print("\nSaved Excel:")
print(excel_output)

print("\n" + "=" * 70)