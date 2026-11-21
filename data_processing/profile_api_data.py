import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/api_data")

csv_files = list(DATA_DIR.glob("*.csv"))

print("=" * 70)
print("API DATASET PROFILING")
print("=" * 70)

print(f"\nTotal CSV files found: {len(csv_files)}")

for file in csv_files:

    print("\n" + "-" * 70)
    print(f"FILE: {file.name}")
    print("-" * 70)

    try:
        df = pd.read_csv(file)

        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")

        print("\nColumns:")
        print(list(df.columns))

        print("\nMissing values:")
        print(df.isnull().sum())

        print(f"\nDuplicate rows: {df.duplicated().sum()}")

        print("\nFirst 2 records:")
        print(df.head(2).to_string(index=False))

    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "=" * 70)
print("PROFILING COMPLETED")
print("=" * 70)