import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
API_DIR = PROJECT_DIR / "data" / "api_data"
FUSION_FILE = PROJECT_DIR / "data" / "fusion" / "InterviewAce_horizontal_fusion.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# API filename prefixes and their corresponding source names
SOURCE_PREFIXES = {
    "github_": "GitHub",
    "hackernews_": "Hacker News",
    "openlibrary_": "Open Library",
    "remotive_": "Remotive",
    "stackexchange_": "Stack Exchange",
}

# --------------------------------------------------
# 1. Inspect original API datasets
# --------------------------------------------------
source_records = []
total_original_records = 0
total_original_files = 0

for file_path in sorted(API_DIR.glob("*.csv")):
    df = pd.read_csv(file_path)
    total_original_records += len(df)
    total_original_files += 1

    source_name = next(
        (
            name
            for prefix, name in SOURCE_PREFIXES.items()
            if file_path.name.startswith(prefix)
        ),
        "Other",
    )

    source_records.append({
        "API_Source": source_name,
        "File_Name": file_path.name,
        "Record_Count": len(df),
        "Column_Count": len(df.columns),
    })

source_df = pd.DataFrame(source_records)
source_df.to_csv(
    OUTPUT_DIR / "original_api_record_counts.csv",
    index=False
)

# Aggregate file counts by API source
source_summary = (
    source_df.groupby("API_Source", as_index=False)
    .agg(
        File_Count=("File_Name", "count"),
        Total_Records=("Record_Count", "sum"),
    )
)

source_summary.to_csv(
    OUTPUT_DIR / "api_source_summary.csv",
    index=False
)

# --------------------------------------------------
# 2. Inspect fused dataset
# --------------------------------------------------
fused_df = pd.read_csv(FUSION_FILE)

fused_rows = len(fused_df)
fused_columns = len(fused_df.columns)

# These are the sources represented in the current
# fused dataset, based on its existing column names.
fused_sources = []

if any(col.startswith("stack_") for col in fused_df.columns):
    fused_sources.append("Stack Exchange")

if any(col.startswith("github_") for col in fused_df.columns):
    fused_sources.append("GitHub")

if any(col.startswith("remotive_") for col in fused_df.columns):
    fused_sources.append("Remotive")

all_sources = set(source_summary["API_Source"])
included_sources = set(fused_sources)
excluded_sources = sorted(all_sources - included_sources)

# --------------------------------------------------
# 3. Create comparison summary
# --------------------------------------------------
comparison = pd.DataFrame([
    {
        "Measure": "Original API CSV files",
        "Value": total_original_files,
    },
    {
        "Measure": "Original API records",
        "Value": total_original_records,
    },
    {
        "Measure": "Original API sources collected",
        "Value": len(all_sources),
    },
    {
        "Measure": "Fused technology records",
        "Value": fused_rows,
    },
    {
        "Measure": "Fused dataset columns",
        "Value": fused_columns,
    },
    {
        "Measure": "Sources represented in fusion",
        "Value": len(included_sources),
    },
    {
        "Measure": "Sources not represented in fusion",
        "Value": len(excluded_sources),
    },
])

comparison.to_csv(
    OUTPUT_DIR / "before_after_fusion_summary.csv",
    index=False
)

# --------------------------------------------------
# 4. Visualization: records collected per API source
# --------------------------------------------------
plt.figure(figsize=(9, 5))
plt.bar(
    source_summary["API_Source"],
    source_summary["Total_Records"]
)
plt.title("Original Records Collected per API Source")
plt.xlabel("API Source")
plt.ylabel("Number of Records")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "original_api_record_counts.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 5. Visualization: popularity score by technology
# --------------------------------------------------
if "technology_popularity_score" in fused_df.columns:
    popularity_df = fused_df.sort_values(
        "technology_popularity_score",
        ascending=False
    )

    plt.figure(figsize=(10, 5))
    plt.bar(
        popularity_df["technology"].astype(str),
        popularity_df["technology_popularity_score"]
    )
    plt.title("Technology Popularity Scores in Fused Dataset")
    plt.xlabel("Technology")
    plt.ylabel("Popularity Score")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "fused_technology_popularity.png",
        dpi=300
    )
    plt.close()

# --------------------------------------------------
# 6. Save a readable terminal report
# --------------------------------------------------
print("\n========== BEFORE VS AFTER FUSION ==========")

print(f"\nOriginal API files: {total_original_files}")
print(f"Original API records: {total_original_records}")
print(f"Original API sources: {len(all_sources)}")

print("\nRecords collected per source:")
print(source_summary.to_string(index=False))

print("\nFused dataset shape:", fused_df.shape)
print("Sources represented in fusion:", sorted(included_sources))
print("Sources not represented in fusion:", excluded_sources)

print("\nFused dataset columns:")
print(fused_df.columns.tolist())

print("\nFiles saved in:", OUTPUT_DIR)
print("1. original_api_record_counts.csv")
print("2. api_source_summary.csv")
print("3. before_after_fusion_summary.csv")
print("4. original_api_record_counts.png")
print("5. fused_technology_popularity.png")

print("\nComparison completed successfully.")