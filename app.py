import tempfile

from click import prompt
from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os

from pypdf import PdfReader

load_dotenv()
app = FastAPI()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
class Document(BaseModel):
    text: str

@app.post("/summarize")
def summarize_document(document: Document):
    prompt = f"""
    Summarize the document.
    Return JSON:
    {{
        "summary": "...",
        "key_points": ["...", "..."]
    }}
    Document:
    {document.text}
    """
    print("Sending request to LLM...")
    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=prompt
    )

    return {"summary": response.output_text}

@app.post("/summarize-txt")
async def summarize_document(file: UploadFile = File(...)):

    content = await file.read()
    text = content.decode("utf-8")

    prompt = f"""
    Summarize the following document.
    Return:
    - summary
    - key bullet points.
    
    Document:
    {text}
    """

    print("Sending request to LLM...")
    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=prompt
    )

    return {"summary": response.output_text}

# pdf text extraction
def extract_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

@app.post("/summarize-pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must end with .pdf")

    try:
        # save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        # extract text from pdf
        document_text = extract_pdf_text(temp_path)

        if not document_text.strip():
            raise HTTPException(status_code=400, detail="could not extract text from pdf file")

        # send text to LLM
        prompt = f"""
        Summarize the following document.
        Return JSON:
        {{
            "summary": "..."
            "key_points": ["...", "..."]
        }}
        
        Document:
        {document_text[:12000]}
        """

        print("Sending request to LLM...")
        response = client.responses.create(
            model="gpt-3.5-turbo",
            input=prompt
        )

        return {"summary": response.output_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
