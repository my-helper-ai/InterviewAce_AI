import os
import fitz  # PyMuPDF
import pandas as pd
from docx import Document


def load_pdf(file_path):
    """Load text from PDF"""
    text = ""

    pdf = fitz.open(file_path)

    for page in pdf:
        text += page.get_text()

    pdf.close()

    return text


def load_txt(file_path):
    """Load text from TXT"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_docx(file_path):
    """Load text from DOCX"""
    doc = Document(file_path)

    text = ""

    for para in doc.paragraphs:
        text += para.text + "\n"

    return text


def load_csv(file_path):
    """Load CSV as text"""
    df = pd.read_csv(file_path)

    return df.to_string(index=False)


def load_document(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    elif extension == ".txt":
        return load_txt(file_path)

    elif extension == ".docx":
        return load_docx(file_path)

    elif extension == ".csv":
        return load_csv(file_path)

    else:
        raise ValueError(f"Unsupported file type: {extension}")