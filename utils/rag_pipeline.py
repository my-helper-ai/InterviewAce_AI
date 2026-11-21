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


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_VECTOR_DB_PATH = PROJECT_ROOT / "vector_db_api"


# --------------------------------------------------
# Interview dataset
# --------------------------------------------------

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


# --------------------------------------------------
# API vector database
# --------------------------------------------------

def load_api_vector_store():
    """Load the FAISS index created from external API data."""

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


# --------------------------------------------------
# Job field extraction
# --------------------------------------------------

def extract_field(content, field):
    """
    Extract a field from a pipe-separated API record.

    Handles both:
        category: Software Development | job_type: contract

    and:
        category: Software Development| job_type: contract
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
    """Extract structured information from a job record."""

    content = document.page_content

    return {
        "job_id": extract_field(content, "job_id"),
        "title": extract_field(content, "title"),
        "company": extract_field(content, "company"),
        "category": extract_field(content, "category"),
        "location": extract_field(content, "location"),
        "job_type": extract_field(content, "job_type")
    }


# --------------------------------------------------
# Job query detection
# --------------------------------------------------

def is_job_question(question):
    """Determine whether a question requests job listings."""

    question_lower = question.lower()

    keywords = [
        "job",
        "jobs",
        "hiring",
        "position",
        "positions",
        "vacancy",
        "vacancies",
        "opening",
        "openings",
        "employment",
        "career opportunities",
        "job opportunities",
        "job listings",
        "job listing"
    ]

    return any(keyword in question_lower for keyword in keywords)


def extract_requested_role(question):
    """
    Extract the requested job role from a natural-language question.
    Supports general roles without restricting them to a predefined list.
    """

    question_clean = question.strip()

    patterns = [
        # Examples: "Find Data Analyst jobs", "List Python Developer jobs"
        r"\b(?:find|show|list|search(?:\s+for)?|get)"
        r"\s+(?:me\s+)?(?:remote\s+)?(.+?)\s+jobs?\b",

        # Examples: "jobs for Data Analysts", "positions in software development"
        r"\b(?:jobs?|positions?|openings?|vacancies|roles?)"
        r"\s+(?:for|in|as)\s+(.+?)(?:[?.!,;]|$)",

        # Examples: "hiring Data Analysts"
        r"\b(?:hiring|recruiting)\s+(.+?)(?:[?.!,;]|$)",

        # Examples: "jobs related to Python"
        r"\b(?:jobs?|positions?)\s+related\s+to\s+"
        r"(.+?)(?:[?.!,;]|$)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            question_clean,
            flags=re.IGNORECASE
        )

        if not match:
            continue

        role = match.group(1).strip()

        # Remove trailing context that is not part of the job role.
        role = re.split(
            r"\b(?:in the collected API data|in the API data|"
            r"in the dataset|in my dataset|from the dataset|"
            r"from the collected data|in collected data)\b",
            role,
            flags=re.IGNORECASE
        )[0].strip()

        # Remove common leading filler.
        role = re.sub(
            r"^(?:some|any|available|remote|the)\s+",
            "",
            role,
            flags=re.IGNORECASE
        ).strip()

        # Remove common trailing filler.
        role = re.sub(
            r"\s+(?:please|available)$",
            "",
            role,
            flags=re.IGNORECASE
        ).strip()

        if role:
            return role

    return None

def normalize_text(text):
    """Normalize text for flexible title and category matching."""

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(text).lower()
    ).strip()


def role_matches_job(role, details):
    """
    Match a requested role against job title and category.
    Supports exact, partial, and carefully defined related-role matches.
    """

    if not role:
        return True

    requested = normalize_text(role)
    title = normalize_text(details.get("title", ""))
    category = normalize_text(details.get("category", ""))

    if not requested:
        return True

    # Exact phrase match.
    if requested in title or requested in category:
        return True

    # Match meaningful words when the full phrase is absent.
    ignored_words = {
        "job", "jobs", "role", "roles", "position", "positions",
        "remote", "available", "opening", "openings", "opportunity",
        "opportunities", "in", "for", "the", "a", "an"
    }

    role_words = [
        word for word in requested.split()
        if word not in ignored_words
    ]

    combined_words = set(f"{title} {category}".split())

    # Require all meaningful words for a direct word-level match.
    if role_words and all(word in combined_words for word in role_words):
        return True

    # Carefully defined role-family matches.
    related_roles = {
        "python developer": {"software developer", "software development"},
        "software developer": {"software development"},
        "data analyst": {"data and analytics"},
        "data scientist": {"data and analytics", "artificial intelligence"},
        "machine learning engineer": {
            "artificial intelligence",
            "software development"
        },
        "cloud engineer": {"cloud computing", "information technology"},
    }

    requested_family = " ".join(requested.split())

    if requested_family in related_roles:
        allowed_categories = related_roles[requested_family]

        return any(
            category == normalize_text(allowed)
            or normalize_text(allowed) in category
            for allowed in allowed_categories
        )

    return False

# --------------------------------------------------
# Dynamic job retrieval
# --------------------------------------------------

def retrieve_job_documents(api_db, question):
    """
    Retrieve all indexed Remotive job records, then filter by role.
    Deduplicate records using job ID.
    """

    requested_role = extract_requested_role(question)

    matching_docs = []
    seen_ids = set()

    try:
        # Inspect all documents stored in the FAISS docstore.
        docstore = api_db.docstore
        all_docs = docstore._dict.values()

    except Exception as error:
        print(f"Could not inspect FAISS docstore: {error}")
        return [], requested_role

    for doc in all_docs:
        source = str(doc.metadata.get("source", "")).strip().lower()

        if source not in {"remotive jobs", "remotive"}:
            continue

        details = get_job_details(doc)

        if details["title"] == "Not provided":
            continue

        if not role_matches_job(requested_role, details):
            continue

        job_id = details["job_id"]

        unique_key = (
            job_id
            if job_id != "Not provided"
            else (
                details["title"].lower(),
                details["company"].lower()
            )
        )

        if unique_key in seen_ids:
            continue

        seen_ids.add(unique_key)
        matching_docs.append(doc)

    return matching_docs, requested_role


# --------------------------------------------------
# Structured job answers
# --------------------------------------------------

def format_job_listing_answer(documents, requested_role=None):
    """Format matching jobs directly from their retrieved source records."""

    if not documents:
        role_description = (
            f" for '{requested_role}'"
            if requested_role
            else ""
        )

        return (
            f"No matching job records{role_description} were found "
            "in the retrieved API data. This does not necessarily mean "
            "that no such jobs exist outside the collected dataset."
        )

    heading = (
        f"Matching job listings for '{requested_role}':"
        if requested_role
        else "Job listings found in the collected API data:"
    )

    lines = [heading, ""]

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
        "Source: Remotive records in the collected API dataset."
    )

    return "\n".join(lines)


# --------------------------------------------------
# Main RAG question-answering pipeline
# --------------------------------------------------

def ask_question(question):
    """Answer questions using interview material and collected API data."""

    interview_db = load_vector_store()
    api_db = load_api_vector_store()

    if interview_db is None and api_db is None:
        return "No interview or API data has been indexed yet.", []

    interview_docs = []
    api_docs = []

    # Retrieve interview-preparation material.
    if interview_db is not None:
        try:
            interview_docs = interview_db.similarity_search(
                question,
                k=3
            )
        except Exception as error:
            print(f"Interview retrieval warning: {error}")

    # Job-listing questions receive structured, source-grounded answers.
    if api_db is not None and is_job_question(question):

        api_docs, requested_role = retrieve_job_documents(
            api_db,
            question
        )

        answer = format_job_listing_answer(
            api_docs,
            requested_role
        )

        return answer, interview_docs + api_docs

    # General questions use normal semantic API retrieval.
    if api_db is not None:
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
4. Do not guess missing company names, locations, dates, or other factual details.
5. If the context is insufficient, state that clearly.
6. For interview questions, explain concepts in an interview-friendly manner.
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