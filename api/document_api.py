from fastapi import FastAPI, UploadFile, File
import tempfile
import os

from utils.rag_pipeline import process_document

app = FastAPI(
    title="InterviewAce AI Document Processing API",
    description="Internal API for processing interview preparation documents.",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "message": "InterviewAce AI Document Processing API is running"
    }


@app.post("/process-document")
async def process_interview_document(
    file: UploadFile = File(...),
    domain: str = "Python"
):

    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        contents = await file.read()
        temp.write(contents)
        temp_path = temp.name

    try:

        process_document(temp_path, domain)

        return {
            "status": "success",
            "message": "Document processed successfully",
            "filename": file.filename,
            "domain": domain
        }

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)