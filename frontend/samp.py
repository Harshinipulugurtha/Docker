#frontend/app.py

import streamlit as st
import requests
from mic_utils import record_and_transcribe 
from tts_utils import speak_text, generate_audio_html
from ner_display import display_ner_highlighted

import os
import torch
from transformers import pipeline
import string
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from io import BytesIO
from pydub import AudioSegment
import tempfile
from PIL import Image, UnidentifiedImageError, Image as PILImage
import json
import requests
from dotenv import load_dotenv
import google.generativeai as genai
import re
import fitz
import io

# Ensure .env is loaded and Gemini API is configured globally
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass

# -------------------------
# Shared Prompt Template for Gemini Analysis
# ------------------------- 
PROMPT_TEMPLATE = """
You are a highly skilled medical expert with specializations across multiple fields like radiology, ophthalmology, and dermatology. Your task is to analyze the provided image or PDF report.

**Instructions:**
1.  **Identify the Image or Report Type:** First, determine the type of medical image or report (e.g., X-ray of a bone, clinical photograph of an eye, dermatological image of a skin condition, lab report, etc.).
2.  **Adopt the Correct Persona:** Based on the type, adopt the appropriate expert role.
    *   For an X-ray, act as a **Radiologist**.
    *   For an eye image, act as an **Ophthalmologist**.
    *   For a skin image, act as a **Dermatologist**.
    *   For a lab report, act as a **General Physician** or relevant specialist.
    *   For other images or reports, use your best judgment to select a relevant medical expert role.
3.  **Provide a Structured Report:** Generate a detailed analysis in a structured format using markdown. The report should include the following sections:
    *   `### Role Adopted:` (State the expert role you have taken on).
    *   `### Observations:` (Describe what you see in the image or report in medical terms).
    *   `### Impression / Potential Diagnosis:` (Provide a potential diagnosis or impression based on the visual evidence).
    *   `### Recommendations:` (Suggest potential next steps, such as specific tests or consultation with a specialist).
4.  **Crucial Disclaimer:** Conclude your analysis with the following mandatory disclaimer, formatted exactly as shown:

---
***Disclaimer:*** This is an AI-generated analysis for educational and informational purposes only. It is **NOT** a substitute for a professional medical diagnosis. Please consult a qualified healthcare provider for any health concerns.*
"""

# -------------------------
# Load Translations
# -------------------------
with open(os.path.join(os.path.dirname(__file__), "assets", "translations.json"), encoding="utf-8") as f:
    TRANSLATIONS = json.load(f)

# -------------------------
# Utility Functions
# -------------------------
def translate_question(text, lang_code):
    if lang_code == "en":
        return text
    model_map = {
        "fr": "Helsinki-NLP/opus-mt-fr-en",
        "es": "Helsinki-NLP/opus-mt-es-en",
        "de": "Helsinki-NLP/opus-mt-de-en",
        "hi": "Helsinki-NLP/opus-mt-hi-en",
        "zh": "Helsinki-NLP/opus-mt-zh-en"
    }
    model_name = model_map.get(lang_code)
    if not model_name:
        return text
    translator = pipeline("translation", model=model_name)
    return translator(text, max_length=512)[0]['translation_text']

def translate_answer(text, lang_code):
    if lang_code == "en" or not text or not isinstance(text, str) or text.strip() == "":
        return text
    model_map = {
        "fr": "Helsinki-NLP/opus-mt-en-fr",
        "es": "Helsinki-NLP/opus-mt-en-es",
        "de": "Helsinki-NLP/opus-mt-en-de",
        "hi": "Helsinki-NLP/opus-mt-en-hi",
        "zh": "Helsinki-NLP/opus-mt-en-zh"
    }
    model_name = model_map.get(lang_code)
    if not model_name:
        return text
    try:
        blocks = re.split(r'(\n+)', text)
        translated_blocks = []
        translator = pipeline("translation", model=model_name)
        for block in blocks:
            if block.strip() and not re.match(r'^\s*#', block):
                translated_block = translator(block, max_length=512)[0]['translation_text']
            else:
                translated_block = block
            translated_blocks.append(translated_block)
        translated = ''.join(translated_blocks)
        lines = translated.splitlines()
        cleaned_lines = []
        prev_line = None
        for line in lines:
            if line.strip() and line != prev_line:
                cleaned_lines.append(line)
            prev_line = line
        cleaned = '\n'.join(cleaned_lines)
        cleaned = re.sub(r'(\b\w+\b)(?:\s+\1){2,}', r'\1', cleaned)
        cleaned = re.sub(r'(\w)\1{4,}', r'\1', cleaned)
        cleaned = cleaned.strip()
        if not cleaned:
            return text
        if lang_code == 'hi' and re.fullmatch(r'[\x00-\x7F\s.,!?\-:;\(\)\[\]"\']*', cleaned):
            return text
        return cleaned
    except Exception as e:
        return f"❌ Translation error: {e}"

def is_greeting(text):
    greetings = {"hi", "hello", "hey", "good morning", "good evening", 
                 "bonjour", "salut", "hola", "hallo", "namaste", "你好"}
    text_clean = text.lower().translate(str.maketrans('', '', string.punctuation)).strip()
    return text_clean in greetings

# -------------------------
# Page Configuration & CSS
# -------------------------
st.set_page_config(page_title=" Medical Assistant", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
        .main-title {
            font-size: 2.5em;
            font-weight: 800;
            color: #003366;
            margin-bottom: 0.5em;
        }
        .section-heading {
            font-size: 1.6em;
            font-weight: 700;
            color: #1a1a1a;
            margin-top: 2em;
        }
        .upload-box {
            border: 2px dashed #ccc;
            padding: 1em;
            border-radius: 12px;
            background-color: #f9f9f9;
        }
        .stDownloadButton>button {
            background-color: #2E8B57 !important;
            color: white !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ... (rest of your code remains unchanged)
