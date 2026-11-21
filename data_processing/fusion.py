import os
import glob
import pandas as pd


API_DATA_FOLDER = "data/api_data"
OUTPUT_FILE = "data/InterviewAce_Data_Fusion.xlsx"


def load_csv_files():

    csv_files = glob.glob(
        os.path.join(API_DATA_FOLDER, "*.csv")
    )

    datasets = {}

    for file in csv_files:

        filename = os.path.basename(file)

        try:
            df = pd.read_csv(file)

            datasets[filename] = df

            print(
                f"Loaded {filename}: {len(df)} records"
            )

        except Exception as e:

            print(
                f"Could not read {filename}: {e}"
            )

    return datasets


def prepare_fused_data(datasets):

    fused_records = []

    for filename, df in datasets.items():

        source = "Unknown"

        if "stackexchange" in filename.lower():
            source = "Stack Exchange"

        elif "github" in filename.lower():
            source = "GitHub"

        elif "hackernews" in filename.lower():
            source = "Hacker News"

        elif "openlibrary" in filename.lower():
            source = "Open Library"

        elif "remotive" in filename.lower():
            source = "Remotive Jobs"

        for _, row in df.iterrows():

            values = []

            for column in df.columns:

                value = row[column]

                if pd.notna(value):

                    values.append(
                        f"{column}: {value}"
                    )

            content = " | ".join(values)

            fused_records.append({
                "source": source,
                "original_file": filename,
                "content": content
            })

    return pd.DataFrame(fused_records)


def create_excel_file(datasets, fused_df):

    os.makedirs(
        "data",
        exist_ok=True
    )

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:

        # Original datasets
        for filename, df in datasets.items():

            sheet_name = (
                os.path.splitext(filename)[0]
            )[:31]

            df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )

        # Final fused dataset
        fused_df.to_excel(
            writer,
            sheet_name="Fused_Data",
            index=False
        )

    print("\nExcel Data Fusion file created:")
    print(OUTPUT_FILE)

    print(
        f"Total fused records: {len(fused_df)}"
    )


if __name__ == "__main__":

    print("\n========== DATA FUSION ==========\n")

    datasets = load_csv_files()

    if not datasets:

        print(
            "No API datasets found."
        )

    else:

        fused_df = prepare_fused_data(
            datasets
        )

        create_excel_file(
            datasets,
            fused_df
        )

        print(
            "\nData Fusion completed successfully!"
        )