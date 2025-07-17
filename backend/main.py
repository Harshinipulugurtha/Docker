#backend/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from model_utils import ask_gemini, analyze_image, highlight_medical_entities, summarize_content
from pdf_utils import extract_text_from_pdf
from pathlib import Path
import os
import os
import shutil
import logging

# ----------------------
# Setup Logging
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# App Configuration
# ----------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set appropriate origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Use a writable data directory relative to the backend folder
DATA_DIR = Path(os.path.join(os.path.dirname(__file__), "../data")).resolve()
DATA_DIR.mkdir(exist_ok=True, parents=True)

# ----------------------
# Ask Endpoint
# ----------------------
@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    context: str = Form(""),
    tone: str = Form("formal"),
    simplify: bool = Form(False),
    role: str = Form("general_physician")
):
    logger.info(f"Received question: {question} | Role: {role}")
    answer = ask_gemini(question, context=context, tone=tone, simple=simplify, role=role)
    highlighted = highlight_medical_entities(answer)
    return {"answer": highlighted}

# ----------------------
# Image Analysis Endpoint
# ----------------------
@app.post("/analyze_image")
async def analyze_image_route(image: UploadFile = File(...), role: str = Form("general_physician")):
    image_path = DATA_DIR / image.filename
    try:
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # Validate image
        from PIL import Image as PILImage
        img = PILImage.open(image_path)
        if img.size[0] < 100 or img.size[1] < 100:
            return {"analysis": "❌ Error: Image too small for analysis. Minimum size is 100x100 pixels."}
        if img.format.upper() not in ["JPEG", "PNG", "JPG"]:
            return {"analysis": f"❌ Error: Unsupported image format: {img.format}. Only JPEG/PNG allowed."}

        result = analyze_image(str(image_path), role=role)
        summary = summarize_content(result, role=role)
        return {"analysis": summary}

    except Exception as e:
        logger.error(f"❌ Error processing image: {e}")
        return {"analysis": f"❌ Error: Could not process image. Reason: {str(e)}"}

# ----------------------
# PDF Upload and Analysis Endpoint
# ----------------------
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), role: str = Form("general_physician")):
    try:
        pdf_path = DATA_DIR / file.filename
        logger.info(f"Saving PDF to: {pdf_path}")

        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        text = extract_text_from_pdf(str(pdf_path))
        logger.info(f"✅ Extracted text preview: {text[:300]}...")

        if not text.strip():
            return {"content": "❌ Error: PDF contains no extractable text."}

        summary = summarize_content(text, role=role)
        return {"content": summary}

    except Exception as e:
        logger.error(f"❌ Error analyzing PDF: {e}")
        return {"content": f"❌ Error processing PDF: {str(e)}"}
