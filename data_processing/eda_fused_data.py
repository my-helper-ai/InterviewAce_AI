import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# PATHS
# =========================================================

INPUT_FILE = Path(
    "data/fusion/InterviewAce_horizontal_fusion.csv"
)

OUTPUT_DIR = Path("data/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nDataset:")
print(df)


# =========================================================
# 1. DESCRIPTIVE STATISTICS
# =========================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

numeric_df = df.select_dtypes(include="number")

print(
    numeric_df.describe().round(2)
)


# =========================================================
# 2. UNIVARIATE ANALYSIS
# DISTRIBUTION OF TOTAL ACTIVITY
# =========================================================

plt.figure(figsize=(9, 6))

plt.hist(
    df["total_activity"],
    bins=6,
    edgecolor="black"
)

plt.xlabel("Total Activity")
plt.ylabel("Number of Technologies")
plt.title(
    "Distribution of Total Technology Activity"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_total_activity_distribution.png",
    dpi=300
)

plt.show()


# =========================================================
# 3. BOX PLOT
# GITHUB STARS
# =========================================================

plt.figure(figsize=(8, 6))

plt.boxplot(
    df["github_avg_stars"]
)

plt.ylabel("Average GitHub Stars")
plt.title(
    "Distribution of Average GitHub Stars"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_github_stars_boxplot.png",
    dpi=300
)

plt.show()


# =========================================================
# 4. RELATIONSHIP PLOT
# STACK EXCHANGE VS GITHUB
# =========================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    df["stack_avg_score"],
    df["github_avg_stars"],
    s=100
)

for _, row in df.iterrows():

    plt.annotate(
        row["technology"],
        (
            row["stack_avg_score"],
            row["github_avg_stars"]
        ),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel("Average Stack Exchange Score")
plt.ylabel("Average GitHub Stars")
plt.title(
    "Stack Exchange Engagement vs GitHub Popularity"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_stack_vs_github.png",
    dpi=300
)

plt.show()


# =========================================================
# 5. RELATIONSHIP PLOT
# JOBS VS COMMUNITY ACTIVITY
# =========================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    df["total_activity"],
    df["remotive_job_count"],
    s=100
)

for _, row in df.iterrows():

    plt.annotate(
        row["technology"],
        (
            row["total_activity"],
            row["remotive_job_count"]
        ),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel("Total Technology Activity")
plt.ylabel("Number of Remotive Jobs")
plt.title(
    "Technology Activity vs Job Availability"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_activity_vs_jobs.png",
    dpi=300
)

plt.show()


# =========================================================
# 6. FUSION-SPECIFIC VISUALIZATION
# TECHNOLOGY COMPARISON
# =========================================================

plot_df = df.sort_values(
    "technology_popularity_score",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    plot_df["technology"],
    plot_df["technology_popularity_score"]
)

plt.xlabel("Technology Popularity Score")
plt.ylabel("Technology")
plt.title(
    "Technology Popularity Based on Fused API Data"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_fused_popularity_comparison.png",
    dpi=300
)

plt.show()


# =========================================================
# 7. CORRELATION HEATMAP
# =========================================================

correlation = numeric_df.corr()


plt.figure(figsize=(14, 11))

plt.imshow(
    correlation,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title(
    "Correlation Heatmap of Fused Dataset"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_correlation_heatmap.png",
    dpi=300
)

plt.show()


# =========================================================
# 8. TOP TECHNOLOGIES
# =========================================================

print("\n" + "=" * 70)
print("TOP TECHNOLOGIES")
print("=" * 70)

top = df.sort_values(
    "technology_popularity_score",
    ascending=False
)

print(
    top[
        [
            "technology",
            "technology_popularity_score",
            "total_activity",
            "remotive_job_count"
        ]
    ].to_string(index=False)
)


# =========================================================
# 9. SAVE EDA SUMMARY
# =========================================================

summary_file = OUTPUT_DIR / "EDA_summary.csv"

df[
    [
        "technology",
        "total_activity",
        "technology_popularity_score",
        "remotive_job_count",
        "github_avg_stars",
        "stack_avg_score"
    ]
].sort_values(
    "technology_popularity_score",
    ascending=False
).to_csv(
    summary_file,
    index=False
)


print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)

print(f"\nEDA outputs saved to:")
print(OUTPUT_DIR)