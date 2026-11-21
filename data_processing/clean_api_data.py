import pandas as pd
from pathlib import Path
import re
import html


# ---------------------------------------------------------
# FOLDERS
# ---------------------------------------------------------

RAW_DIR = Path("data/api_data")
CLEAN_DIR = Path("data/cleaned")

CLEAN_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# TECHNOLOGY NORMALIZATION
# ---------------------------------------------------------

TECH_MAP = {
    "python": "Python",
    "python-programming": "Python",

    "sql": "SQL",
    "sql-server": "SQL",

    "pandas": "Pandas",

    "machine-learning": "Machine Learning",
    "machine learning": "Machine Learning",

    "data-analysis": "Data Analysis",
    "data analysis": "Data Analysis",

    "devops": "DevOps",
    "cloud": "Cloud Computing",
    "cloud-computing": "Cloud Computing",

    "customer-support": "Customer Support",
    "customer support": "Customer Support",

    "software-development": "Software Development",
}


def normalize_technology(value):

    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().lower()

    return TECH_MAP.get(value, value.title())


# ---------------------------------------------------------
# CLEAN TEXT
# ---------------------------------------------------------

def clean_text(value):

    if pd.isna(value):
        return ""

    value = html.unescape(str(value))

    # Remove HTML tags
    value = re.sub(r"<[^>]+>", " ", value)

    # Remove excessive whitespace
    value = re.sub(r"\s+", " ", value)

    return value.strip()


# ---------------------------------------------------------
# STACK EXCHANGE
# ---------------------------------------------------------

def clean_stackexchange(df):

    df = df.copy()

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Remove duplicate questions using question_id
    if "question_id" in df.columns:
        df = df.drop_duplicates(subset=["question_id"])

    # Clean text columns
    for col in ["title", "tags", "link"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Numeric columns
    for col in ["score", "answer_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Normalize domain
    if "domain" in df.columns:
        df["technology"] = df["domain"].apply(normalize_technology)

    return df


# ---------------------------------------------------------
# GITHUB
# ---------------------------------------------------------

def clean_github(df):

    df = df.copy()

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Repository should uniquely identify a repository
    if "repository" in df.columns:
        df = df.drop_duplicates(subset=["repository"])

    # Clean text
    for col in ["topic", "repository", "description", "language", "url"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Numeric columns
    for col in ["stars", "forks"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Create common technology key
    if "topic" in df.columns:
        df["technology"] = df["topic"].apply(normalize_technology)

    return df


# ---------------------------------------------------------
# REMOTIVE
# ---------------------------------------------------------

def detect_remotive_technology(row):

    title = str(row.get("title", "")).lower()
    category = str(row.get("category", "")).lower()
    description = str(row.get("description", "")).lower()

    text = f"{title} {category} {description}"

    # Check specific technologies first
    if "machine learning" in text or "machine-learning" in text:
        return "Machine Learning"

    if "python" in text:
        return "Python"

    if "pandas" in text:
        return "Pandas"

    if "sql" in text:
        return "SQL"

    if "devops" in text:
        return "DevOps"

    if "cloud" in text or "aws" in text or "azure" in text:
        return "Cloud Computing"

    if "data analyst" in text or "data analysis" in text:
        return "Data Analysis"

    # Fall back to job category
    if "customer support" in category:
        return "Customer Support"

    if "software development" in category:
        return "Software Development"

    return "Other"


def clean_remotive(df):

    df = df.copy()

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Remove duplicate jobs
    if "job_id" in df.columns:
        df = df.drop_duplicates(subset=["job_id"])

    # Clean text columns
    text_columns = [
        "title",
        "company",
        "category",
        "job_type",
        "location",
        "description",
        "url"
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Create common technology key
    df["technology"] = df.apply(
        detect_remotive_technology,
        axis=1
    )

    return df


# ---------------------------------------------------------
# MAIN PROCESS
# ---------------------------------------------------------

def main():

    files = list(RAW_DIR.glob("*.csv"))

    print("=" * 70)
    print("API DATA CLEANING")
    print("=" * 70)

    print(f"\nRaw CSV files found: {len(files)}")

    for file in files:

        print("\n" + "-" * 70)
        print(f"Processing: {file.name}")
        print("-" * 70)

        try:

            df = pd.read_csv(file)

            original_rows = len(df)

            # Identify API type from columns
            columns = set(df.columns)

            if "question_id" in columns:
                df = clean_stackexchange(df)
                api_type = "Stack Exchange"

            elif "repository" in columns:
                df = clean_github(df)
                api_type = "GitHub"

            elif "job_id" in columns:
                df = clean_remotive(df)
                api_type = "Remotive"

            else:
                print("Unknown dataset format. Skipping.")
                continue

            # Save cleaned copy
            output_file = CLEAN_DIR / file.name

            df.to_csv(
                output_file,
                index=False,
                encoding="utf-8-sig"
            )

            print(f"API Type       : {api_type}")
            print(f"Original rows  : {original_rows}")
            print(f"Cleaned rows   : {len(df)}")
            print(f"Rows removed   : {original_rows - len(df)}")
            print(f"Saved to       : {output_file}")

            if "technology" in df.columns:
                print("\nTechnology distribution:")
                print(df["technology"].value_counts().head(10))

        except Exception as e:

            print(f"ERROR processing {file.name}: {e}")

    print("\n" + "=" * 70)
    print("CLEANING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()