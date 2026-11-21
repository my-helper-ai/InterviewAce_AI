import pandas as pd
import os


EXCEL_FILE = "data/InterviewAce_Data_Fusion.xlsx"


def load_fused_data():

    print("\nLoading fused API dataset...")

    if not os.path.exists(EXCEL_FILE):
        raise FileNotFoundError(
            f"File not found: {EXCEL_FILE}"
        )

    df = pd.read_excel(
        EXCEL_FILE,
        sheet_name="Fused_Data"
    )

    print(
        f"Loaded {len(df)} fused records."
    )

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSources:")
    print(df["source"].value_counts())

    return df


def convert_to_documents(df):

    documents = []

    for _, row in df.iterrows():

        source = row.get(
            "source",
            "Unknown"
        )

        original_file = row.get(
            "original_file",
            "Unknown"
        )

        content = row.get(
            "content",
            ""
        )

        if pd.isna(content):
            continue

        text = (
            f"Source: {source}\n"
            f"Dataset File: {original_file}\n"
            f"Content: {content}"
        )

        documents.append(text)

    return documents


if __name__ == "__main__":

    df = load_fused_data()

    documents = convert_to_documents(df)

    print(
        f"\nConverted {len(documents)} "
        "records into text documents."
    )

    print("\nSample document:\n")

    if documents:
        print(documents[0][:1000])