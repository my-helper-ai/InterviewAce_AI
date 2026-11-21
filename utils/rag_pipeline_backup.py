import csv
import re
from pathlib import Path

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

from utils.loader import load_document
from utils.splitter import split_text
from utils.vector_db import create_vector_store, load_vector_store
from utils.embeddings import load_embedding_model
from config import LLM_MODEL


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_VECTOR_DB_PATH = PROJECT_ROOT / "vector_db_api"


def load_interview_dataset(domain=None):
    """Load interview questions and answers from the local dataset."""

    dataset_path = PROJECT_ROOT / "data" / "interview_dataset.csv"

    if not dataset_path.exists():
        return ""

    records = []

    with open(dataset_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if domain and row.get("domain") != domain:
                continue

            records.append(
                f"Domain: {row.get('domain', '')}\n"
                f"Topic: {row.get('topic', '')}\n"
                f"Difficulty: {row.get('difficulty', '')}\n"
                f"Question: {row.get('question', '')}\n"
                f"Answer: {row.get('answer', '')}"
            )

    return "\n\n".join(records)


def process_document(file_path, domain=None):
    """Process an uploaded document and index it with interview material."""

    text = load_document(file_path)
    dataset_text = load_interview_dataset(domain)

    combined_text = dataset_text + "\n\n" + text
    chunks = split_text(combined_text)

    create_vector_store(chunks)


def load_api_vector_store():
    """Load the separate FAISS index built from external API data."""

    if not API_VECTOR_DB_PATH.exists():
        return None

    if not (API_VECTOR_DB_PATH / "index.faiss").exists():
        return None

    embeddings = load_embedding_model()

    return FAISS.load_local(
        str(API_VECTOR_DB_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )


def extract_job_field(content, field):
    """
    Extract a field from pipe-separated API records.

    Supports fields appearing after either a pipe or the beginning
    of the record, including records without spaces around separators.
    """

    pattern = (
        rf"(?:^|\|)\s*{re.escape(field)}\s*:\s*"
        rf"(.*?)"
        rf"(?=\s*\|\s*[A-Za-z_][A-Za-z0-9_ ]*\s*:|$)"
    )

    match = re.search(
        pattern,
        content,
        flags=re.IGNORECASE | re.DOTALL
    )

    if match:
        value = match.group(1).strip()

        if value:
            return value

    return "Not provided"


def get_job_details(document):
    """Extract structured fields from a Remotive job record."""

    content = document.page_content

    return {
        "job_id": extract_job_field(content, "job_id"),
        "title": extract_job_field(content, "title"),
        "company": extract_job_field(content, "company"),
        "category": extract_job_field(content, "category"),
        "location": extract_job_field(content, "location"),
        "job_type": extract_job_field(content, "job_type")
    }


def is_job_question(question):
    """Identify questions asking for job listings."""

    question_lower = question.lower()

    keywords = [
        "job", "jobs", "hiring", "position", "positions",
        "vacancy", "vacancies", "opening", "openings",
        "employment"
    ]

    return any(keyword in question_lower for keyword in keywords)


def is_software_development_job_question(question):
    """Identify job queries specifically about software development."""

    question_lower = question.lower()

    keywords = [
        "software development",
        "software developer",
        "software engineering",
        "software engineer",
        "programming job",
        "programming jobs",
        "developer job",
        "developer jobs"
    ]

    return any(keyword in question_lower for keyword in keywords)


def retrieve_job_documents(api_db, question):
    """
    Retrieve and filter Remotive job records.

    The category is extracted from each record rather than relying
    on semantic similarity alone.
    """

    try:
        candidate_docs = api_db.similarity_search(
            "Remotive job title company category location job_id",
            k=50,
            filter={"source": "Remotive Jobs"}
        )
    except Exception as error:
        print(f"Metadata-filter retrieval warning: {error}")

        candidate_docs = api_db.similarity_search(
            "Remotive job title company category location",
            k=50
        )

    software_only = is_software_development_job_question(question)

    matching_docs = []
    seen_ids = set()

    for doc in candidate_docs:

        # Confirm the document comes from the Remotive API index.
        source = str(doc.metadata.get("source", "")).strip().lower()

        if source not in {"remotive jobs", "remotive"}:
            continue

        details = get_job_details(doc)

        if details["title"] == "Not provided":
            continue

        # Restrict results to the requested category.
        if software_only:
            category = details["category"].lower().strip()
            normalized_category = re.sub(r"[\s_-]+", "", category)

            if normalized_category != "softwaredevelopment":
                continue

        # Remove duplicate job records using the job ID.
        job_id = details["job_id"]

        if job_id != "Not provided":
            if job_id in seen_ids:
                continue

            seen_ids.add(job_id)
        else:
            # Fallback deduplication when a job ID is unavailable.
            fallback_key = (
                details["title"].strip().lower(),
                details["company"].strip().lower()
            )

            if fallback_key in seen_ids:
                continue

            seen_ids.add(fallback_key)

        matching_docs.append(doc)

    return matching_docs


def format_job_listing_answer(documents):
    """Format job listings directly from the source records."""

    if not documents:
        return (
            "No matching software development job records were found "
            "in the retrieved API data."
        )

    lines = [
        "Software development jobs found in the collected API data:\n"
    ]

    for index, doc in enumerate(documents, start=1):
        details = get_job_details(doc)

        lines.extend([
            f"{index}. Job Title: {details['title']}",
            f"   Company: {details['company']}",
            f"   Category: {details['category']}",
            f"   Location: {details['location']}",
            f"   Job Type: {details['job_type']}",
            ""
        ])

    lines.append(
        "Source: Retrieved Remotive records from the collected API dataset."
    )

    return "\n".join(lines)


def ask_question(question):
    """Retrieve context and answer using interview and API data."""

    interview_db = load_vector_store()
    api_db = load_api_vector_store()

    if interview_db is None and api_db is None:
        return "No interview or API data has been indexed yet.", []

    interview_docs = []
    api_docs = []

    # Retrieve interview-preparation context.
    if interview_db is not None:
        try:
            interview_docs = interview_db.similarity_search(
                question,
                k=3
            )
        except Exception as error:
            print(f"Interview retrieval warning: {error}")

    # Retrieve external API context.
    if api_db is not None:

        if is_job_question(question):
            api_docs = retrieve_job_documents(api_db, question)

            # Use exact source fields for structured job listings.
            if is_software_development_job_question(question):
                answer = format_job_listing_answer(api_docs)
                return answer, interview_docs + api_docs

        else:
            try:
                api_docs = api_db.similarity_search(
                    question,
                    k=6
                )
            except Exception as error:
                print(f"API retrieval warning: {error}")

    interview_context = "\n\n".join(
        doc.page_content for doc in interview_docs
    )

    api_context = "\n\n".join(
        doc.page_content for doc in api_docs
    )

    prompt = PromptTemplate(
        input_variables=[
            "interview_context",
            "api_context",
            "question"
        ],
        template="""You are InterviewAce AI, an expert interview preparation assistant.

Answer the user's question using the retrieved context.

INTERVIEW PREPARATION CONTEXT:
{interview_context}

API-DERIVED CONTEXT:
{api_context}

QUESTION:
{question}

INSTRUCTIONS:
1. Answer clearly using the retrieved information.
2. Distinguish interview-preparation information from API-derived information.
3. Do not invent facts, statistics, job listings, repository details, or book information.
4. Do not guess missing company names, locations, dates, or other facts.
5. If the context is insufficient, state that clearly.
6. Explain interview concepts in an interview-friendly manner.
7. Base factual claims on the supplied context.

ANSWER:
"""
    )

    final_prompt = prompt.format(
        interview_context=(
            interview_context or "No interview context retrieved."
        ),
        api_context=(
            api_context or "No API context retrieved."
        ),
        question=question
    )

    llm = OllamaLLM(model=LLM_MODEL)
    response = llm.invoke(final_prompt)

    # Preserve the return format expected by app.py.
    retrieved_docs = interview_docs + api_docs

    return response, retrieved_docs