from fastapi import FastAPI
from pydantic import BaseModel

from utils.rag_pipeline import ask_question


app = FastAPI(
    title="InterviewAce AI Question Answer API",
    description="RAG-based interview question answering API.",
    version="1.0"
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "InterviewAce AI Question Answer API is running"
    }


@app.post("/ask")
def ask_interview_question(request: QuestionRequest):

    if not request.question.strip():
        return {
            "status": "error",
            "message": "Question cannot be empty"
        }

    answer, docs = ask_question(request.question)

    sources = []

    for doc in docs:
        sources.append(doc.page_content)

    return {
        "status": "success",
        "question": request.question,
        "answer": answer,
        "retrieved_chunks": sources
    }