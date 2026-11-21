import pandas as pd
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
FUSION_FILE = (
    PROJECT_DIR
    / "data"
    / "fusion"
    / "InterviewAce_horizontal_fusion.csv"
)
OUTPUT_DIR = PROJECT_DIR / "data" / "insights"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load fused dataset
df = pd.read_csv(FUSION_FILE)

# Ensure numerical columns contain numeric values
numeric_columns = [
    "technology_popularity_score",
    "github_repository_count",
    "stack_question_count",
    "remotive_job_count",
    "github_total_stars",
    "stack_avg_score",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

insights = []

def add_insight(title, technology, value, explanation):
    insights.append({
        "Insight": title,
        "Technology": technology,
        "Observed_Value": value,
        "Interpretation": explanation,
    })

# 1. Highest popularity score
row = df.loc[df["technology_popularity_score"].idxmax()]

add_insight(
    "Highest technology popularity score",
    row["technology"],
    row["technology_popularity_score"],
    "This technology has the highest calculated popularity score "
    "within the current fused dataset. The score is a constructed "
    "metric, not an independently validated popularity measure.",
)

# 2. Highest GitHub repository count
row = df.loc[df["github_repository_count"].idxmax()]

add_insight(
    "Highest GitHub repository count",
    row["technology"],
    row["github_repository_count"],
    "This technology has the largest repository count in the "
    "collected GitHub data represented by the fused dataset.",
)

# 3. Highest Stack Exchange question count
row = df.loc[df["stack_question_count"].idxmax()]

add_insight(
    "Highest Stack Exchange question count",
    row["technology"],
    row["stack_question_count"],
    "This technology has the highest question count in the "
    "Stack Exchange data represented by the fused dataset.",
)

# 4. Highest number of related jobs
row = df.loc[df["remotive_job_count"].idxmax()]

add_insight(
    "Highest related job count",
    row["technology"],
    row["remotive_job_count"],
    "This technology has the highest number of related job "
    "listings in the collected Remotive data. This does not "
    "represent the entire job market.",
)

# 5. Relationship between community and repository activity
valid = df[
    ["stack_question_count", "github_repository_count"]
].dropna()

if (
    len(valid) >= 2
    and valid["stack_question_count"].nunique() > 1
    and valid["github_repository_count"].nunique() > 1
):
    correlation = valid["stack_question_count"].corr(
        valid["github_repository_count"]
    )

    if correlation >= 0.5:
        interpretation = (
            "The two variables show a positive association "
            "in this small dataset."
        )
    elif correlation <= -0.5:
        interpretation = (
            "The two variables show a negative association "
            "in this small dataset."
        )
    else:
        interpretation = (
            "The association is weak or limited in this dataset."
        )

    add_insight(
        "Community questions vs GitHub repositories",
        "Multiple technologies",
        round(correlation, 3),
        interpretation + " Correlation does not establish causation.",
    )
else:
    add_insight(
        "Community questions vs GitHub repositories",
        "Multiple technologies",
        "Not calculated",
        "There is insufficient variation or valid data to "
        "calculate a meaningful correlation.",
    )

# Save insights
insights_df = pd.DataFrame(insights)

output_file = OUTPUT_DIR / "generated_insights.csv"
insights_df.to_csv(output_file, index=False)

# Display results
print("\n========== GENERATED DATA INSIGHTS ==========")
print(insights_df.to_string(index=False))
print(f"\nInsights saved to: {output_file}")