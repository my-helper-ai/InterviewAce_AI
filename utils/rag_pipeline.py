from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from utils.loader import load_document
from utils.splitter import split_text
from utils.vector_db import create_vector_store, load_vector_store

from config import LLM_MODEL


# ---------------------------------------------------
# Process Uploaded Document
# ---------------------------------------------------

def process_document(file_path):
    """
    Load the uploaded document, split it into chunks,
    generate embeddings and store them in FAISS.
    """

    text = load_document(file_path)

    chunks = split_text(text)

    create_vector_store(chunks)


# ---------------------------------------------------
# Ask Question
# ---------------------------------------------------

def ask_question(question, domain="General", language="English"):

    vector_db = load_vector_store()

    if vector_db is None:
        return "Please upload and process a document first.", []

    # Retrieve relevant chunks
    docs = vector_db.similarity_search(
        question,
        k=5
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = PromptTemplate.from_template(
        """
You are InterviewAce AI.

Role:
You are an intelligent interview preparation assistant.

Interview Domain:
{domain}

Instructions:

1. Answer ONLY from the provided context.

2. If the answer is unavailable in the context,
politely say:

"I could not find this information in the uploaded interview material."

3. Keep answers interview-oriented.

4. Explain concepts clearly.

5. Provide examples whenever possible.

6. If the user asks:
- Summarize
- Explain this document
- Overview

then provide a concise summary of the retrieved content.

7. Respond completely in:
{language}

Context:
{context}

Question:
{question}

Answer:
"""
    )

    final_prompt = prompt.format(
        domain=domain,
        language=language,
        context=context,
        question=question
    )

    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=0.3
    )

    response = llm.invoke(final_prompt)

    return response, docs


# ---------------------------------------------------
# Generate Interview Notes
# ---------------------------------------------------

def generate_notes(topic):

    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=0.4
    )

    prompt = f"""
Generate concise interview revision notes on:

{topic}

Format:

1. Definition

2. Important Concepts

3. Key Points

4. Frequently Asked Interview Questions

5. Quick Revision Tips

Keep the notes easy to revise before an interview.
"""

    notes = llm.invoke(prompt)

    return notes