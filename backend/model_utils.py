# backend/model_utils.py
import google.generativeai as genai
from transformers import pipeline
import os
from PIL import Image

genai.configure(api_key=os.getenv("API_KEY"))
NER = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

def ask_gemini(question, context="", tone="formal", simple=False, role="general_physician"):
    tone_map = {
        "formal": "Give a detailed medical explanation.",
        "friendly": "Respond warmly and clearly.",
        "child": "Explain like to a 10-year-old."
    }
    role_map = {
        "radiologist": "You are a radiologist (a doctor who looks at medical images like X-rays).",
        "general_physician": "You are a general physician (a doctor for general health problems).",
        "orthopedist": "You are an orthopedist (a doctor who treats bone and joint issues).",
        "cardiologist": "You are a cardiologist (a doctor who specializes in heart and blood vessels).",
        "neurologist": "You are a neurologist (a doctor who treats brain and nervous system disorders).",
        "dermatologist": "You are a dermatologist (a doctor who treats skin conditions).",
        "pediatrician": "You are a pediatrician (a doctor who treats children).",
        "dentist": "You are a dentist (a doctor who treats teeth and oral health)."
    }
    # Check for image/PDF analysis in context and instruct model to answer as a doctor
    analysis_in_context = "Image Analysis:" in context or "PDF Analysis:" in context
    role_instruction = role_map.get(role, "You are a helpful medical assistant.")
    if analysis_in_context:
        prompt = f"""{role_instruction}\nUse the following analysis results to answer the user's question in a clear, professional, and explanatory way. Reference the analysis and explain your reasoning as a doctor would.
Context: {context}
Question: {question}
Tone: {tone_map.get(tone, '')}

Answer:"""
    elif simple:
        prompt = f"""{role_instruction}\nContext: {context}\nQuestion: {question}\nPlease answer in 5-8 sentences, using simple words, as if explaining to a young child or someone with no medical background. The answer should be moderate in length, not too short and not too long. Avoid technical jargon and greetings.\n\nAnswer:"""
    else:
        prompt = f"""{role_instruction}\nContext: {context}\nQuestion: {question}\nTone: {tone_map.get(tone, '')}\n\nAnswer:"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

def analyze_image(image_path, role="radiologist"):
    role_map = {
        "radiologist": "You are a radiologist (a doctor who looks at medical images like X-rays).",
        "general_physician": "You are a general physician (a doctor for general health problems).",
        "orthopedist": "You are an orthopedist (a doctor who treats bone and joint issues).",
        "cardiologist": "You are a cardiologist (a doctor who specializes in heart and blood vessels).",
        "neurologist": "You are a neurologist (a doctor who treats brain and nervous system disorders).",
        "dermatologist": "You are a dermatologist (a doctor who treats skin conditions).",
        "pediatrician": "You are a pediatrician (a doctor who treats children).",
        "dentist": "You are a dentist (a doctor who treats teeth and oral health)."
    }
    role_instruction = role_map.get(role, "You are a helpful medical assistant.")
    prompt = f"{role_instruction} Analyze the medical image for abnormalities."
    model = genai.GenerativeModel("gemini-2.5-flash")
    img = Image.open(image_path)
    response = model.generate_content([prompt, img])
    return response.text.strip()

def highlight_medical_entities(text)                                                                                        :
    entities = NER(text)
    for ent in entities:
        label = ent['entity_group']
        word = ent['word']
        emoji = {"DISEASE": "🦠", "SYMPTOM": "🤒", "MEDICATION": "💊"}.get(label.upper(), "🔍")
        text = text.replace(word, f"**{emoji} {word}**")
    return text

def summarize_content(content, role="general_physician"):
    role_map = {
        "radiologist": "You are a radiologist (a doctor who looks at medical images like X-rays).",
        "general_physician": "You are a general physician (a doctor for general health problems).",
        "orthopedist": "You are an orthopedist (a doctor who treats bone and joint issues).",
        "cardiologist": "You are a cardiologist (a doctor who specializes in heart and blood vessels).",
        "neurologist": "You are a neurologist (a doctor who treats brain and nervous system disorders).",
        "dermatologist": "You are a dermatologist (a doctor who treats skin conditions).",
        "pediatrician": "You are a pediatrician (a doctor who treats children).",
        "dentist": "You are a dentist (a doctor who treats teeth and oral health)."
    }
    role_instruction = role_map.get(role, "You are a helpful medical assistant.")
    prompt = (
        f"{role_instruction} Summarize the following medical report in simple, clear language. "
        "Make it easy to understand for non-experts, avoid technical jargon, "
        "and keep the summary moderately detailed (about 5-8 sentences, not too short, not too long). "
        "Do not include greetings.\n\nReport:\n"
        f"{content}\n\nSummary:"
    )
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()
