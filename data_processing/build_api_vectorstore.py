import os
import pandas as pd

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


EXCEL_FILE = "data/InterviewAce_Data_Fusion.xlsx"
OUTPUT_DIR = "vector_db_api"


print("\nLoading fused API data...")

df = pd.read_excel(
    EXCEL_FILE,
    sheet_name="Fused_Data"
)

print(f"Loaded {len(df)} records.")

# Remove duplicate Remotive jobs while preserving all other API records.
remotive_mask = df["source"].astype(str).str.contains(
    "Remotive", case=False, na=False
)

remotive = df[remotive_mask].copy()
other_sources = df[~remotive_mask].copy()

# Extract job IDs from Remotive content.
remotive["job_id"] = remotive["content"].astype(str).str.extract(
    r"job_id:\s*(\d+)"
)

# Deduplicate by job ID; retain the first occurrence.
remotive_with_id = remotive[remotive["job_id"].notna()]
remotive_without_id = remotive[remotive["job_id"].isna()]

remotive_clean = pd.concat(
    [
        remotive_with_id.drop_duplicates(subset=["job_id"]),
        remotive_without_id
    ],
    ignore_index=True
).drop(columns=["job_id"])

df = pd.concat(
    [other_sources, remotive_clean],
    ignore_index=True
)

print(f"Records after deduplication: {len(df)}")
print(f"Remotive records retained: {len(remotive_clean)}")

documents = []

for _, row in df.iterrows():

    content = str(row["content"])

    metadata = {
        "source": str(row["source"]),
        "file": str(row["original_file"]),
        "type": "api_data"
    }

    documents.append(
        Document(
            page_content=content,
            metadata=metadata
        )
    )


print(f"Created {len(documents)} documents.")


print("\nCreating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


print("Building FAISS vector store...")

vectorstore = FAISS.from_documents(
    documents,
    embeddings
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

vectorstore.save_local(
    OUTPUT_DIR
)


print("\n================================")
print("API VECTOR STORE CREATED")
print("================================")
print(f"Records indexed: {len(documents)}")
print(f"Saved to: {OUTPUT_DIR}")