import os

from langchain_community.vectorstores import FAISS

from utils.embeddings import load_embedding_model
from config import VECTOR_DB_PATH


def create_vector_store(chunks):
    """
    Create and save FAISS vector database
    """

    embeddings = load_embedding_model()

    vector_db = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    vector_db.save_local(VECTOR_DB_PATH)

    return vector_db


def load_vector_store():
    """
    Load existing vector database
    """

    embeddings = load_embedding_model()

    if not os.path.exists(VECTOR_DB_PATH):
        return None

    vector_db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_db