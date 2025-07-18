# import streamlit as st
# from PIL import Image
# import google.generativeai as genai
# import time
# import io
# import os  # <<< CHANGE HERE: Import the os library
# from dotenv import load_dotenv  # <<< CHANGE HERE: Import the dotenv library

# # --- Load Environment Variables ---
# load_dotenv()  # <<< CHANGE HERE: This loads the variables from your .env file

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Dynamic AI Medical Analyzer",
#     page_icon="🔬",
#     layout="wide"
# )

# # --- Google Gemini API Configuration ---
# # <<< ENTIRE BLOCK MODIFIED TO USE OS.GETENV >>>
# try:
#     # Configure the API key from environment variables
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         raise ValueError("GOOGLE_API_KEY not found. Please check your .env file.")
#     genai.configure(api_key=api_key)
# except Exception as e:
#     st.error(f"Error configuring Google API: {e}")
#     st.error("Please make sure you have a GOOGLE_API_KEY in your .env file in the root directory.")
#     st.stop()
# # <<< END OF MODIFIED BLOCK >>>


# # --- AI Model and Prompting Function ---
# def get_gemini_analysis(image_bytes, prompt):
#     """
#     Sends the image and prompt to the Gemini Pro Vision model
#     and returns the generated analysis.
#     """
#     try:
#         # Create an Image object from the bytes
#         image_part = Image.open(io.BytesIO(image_bytes))
        
#         # Instantiate the model
#         model = genai.GenerativeModel('gemini-1.5-flash')
        
#         # The content must be a list: [prompt, image]
#         response = model.generate_content([prompt, image_part])
        
#         # Return the text part of the response
#         return response.text
#     except Exception as e:
#         return f"An error occurred during AI analysis: {e}"


# # --- The Master Prompt ---
# # This prompt instructs the AI on how to behave.
# # It tells it to adopt a persona based on the image content.
# PROMPT_TEMPLATE = """
# You are a highly skilled medical expert with specializations across multiple fields like radiology, ophthalmology, and dermatology. Your task is to analyze the provided image.

# **Instructions:**
# 1.  **Identify the Image Type:** First, determine the type of medical image (e.g., X-ray of a bone, clinical photograph of an eye, dermatological image of a skin condition, etc.).
# 2.  **Adopt the Correct Persona:** Based on the image type, adopt the appropriate expert role.
#     *   For an X-ray, act as a **Radiologist**.
#     *   For an eye image, act as an **Ophthalmologist**.
#     *   For a skin image, act as a **Dermatologist**.
#     *   For other images, use your best judgment to select a relevant medical expert role.
# 3.  **Provide a Structured Report:** Generate a detailed analysis in a structured format using markdown. The report should include the following sections:
#     *   `### Role Adopted:` (State the expert role you have taken on).
#     *   `### Observations:` (Describe what you see in the image in medical terms).
#     *   `### Impression / Potential Diagnosis:` (Provide a potential diagnosis or impression based on the visual evidence).
#     *   `### Recommendations:` (Suggest potential next steps, such as specific tests or consultation with a specialist).
# 4.  **Crucial Disclaimer:** Conclude your analysis with the following mandatory disclaimer, formatted exactly as shown:

# ---
# ***Disclaimer:** This is an AI-generated analysis for educational and informational purposes only. It is **NOT** a substitute for a professional medical diagnosis. Please consult a qualified healthcare provider for any health concerns.*
# """

# # --- Streamlit App UI ---
# st.title("🔬 Dynamic AI Medical Image Analyzer")
# st.markdown("Upload any medical image, and the AI will adopt the appropriate expert role to provide an analysis.")

# st.warning(
#     "**For Educational Use Only.** This tool uses a general-purpose AI and is not a certified medical device. "
#     "Do not use it for self-diagnosis. Always consult a real doctor."
# )

# # --- Image Uploader and Analysis Logic ---
# uploaded_file = st.file_uploader("Upload a medical image to analyze...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Get the image bytes
#     image_bytes = uploaded_file.getvalue()

#     # Display the uploaded image and the analysis in two columns
#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("Uploaded Image")
#         st.image(image_bytes, caption="Image ready for analysis", use_column_width=True)

#     with col2:
#         st.subheader("AI Analysis")
#         # Analyze button
#         if st.button("Analyze Image", type="primary"):
#             with st.spinner('The AI expert is examining the image... Please wait.'):
#                 # Call the AI function
#                 analysis_text = get_gemini_analysis(image_bytes, PROMPT_TEMPLATE)
                
#                 # Display the result
#                 st.markdown(analysis_text)






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
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
import google.generativeai as genai
import json
import requests

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
        result = pipeline("translation", model=model_name)(text, max_length=512)
        if result and isinstance(result, list) and 'translation_text' in result[0]:
            return result[0]['translation_text']
        else:
            return text
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

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("🩺 Medical Assistant")
if "last_page" not in st.session_state:
    st.session_state.last_page = "🏞️ Home"
page = st.sidebar.radio("Navigate", [
    "🏞️ Home", "🖼️ Image Analysis", "📄 PDF Report Analysis", "🎙️ Voice Q&A", "💬 Chatbot Q&A", "🩹 Symptom Bot"],
    index=["🏞️ Home", "🖼️ Image Analysis", "📄 PDF Report Analysis", "🎙️ Voice Q&A", "💬 Chatbot Q&A", "🩹 Symptom Bot"].index(st.session_state.last_page) if "last_page" in st.session_state else 0)
st.session_state.last_page = page


# -------------------------
# Session State Initialization
# -------------------------
for key in ["image_messages", "pdf_messages", "messages"]:
    if key not in st.session_state:
        st.session_state[key] = []
for key in ["image_analysis", "pdf_analysis"]:
    if key not in st.session_state:
        st.session_state[key] = ""
for key in ["image_analysis_spoken", "pdf_analysis_spoken"]:
    if key not in st.session_state:
        st.session_state[key] = False
for key in ["image_analysis_displayed", "pdf_analysis_displayed"]:
    if key not in st.session_state:
        st.session_state[key] = False


BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# -------------------------
# Global Inputs (Language & Role)
# -------------------------


language_map = {
    "en": "English", "fr": "French", "es": "Spanish",
    "de": "German", "hi": "Hindi", "zh": "Chinese"
}



output_lang = st.sidebar.selectbox(
    "🌐 " + TRANSLATIONS["en"]["language_selection"],
    list(language_map.keys()),
    format_func=lambda x: language_map[x]
)
input_lang = st.sidebar.selectbox(
    "🎙️ Voice input language:",
    list(language_map.keys()),
    format_func=lambda x: language_map[x]
)
simple_explanation = st.sidebar.checkbox("📖 Simple explanation mode (for kids / non-experts)")
tone = st.sidebar.selectbox("🧘 Tone:", ["formal", "friendly", "child"])

role_map = {
    "radiologist": "Radiologist",
    "general_physician": "General Physician",
    "orthopedist": "Orthopedist",
    "cardiologist": "Cardiologist",
    "neurologist": "Neurologist",
    "dermatologist": "Dermatologist",
    "pediatrician": "Pediatrician",
    "dentist": "Dentist"
}
role = st.sidebar.selectbox("👨‍⚕️ Medical Expert Role:", list(role_map.keys()), format_func=lambda x: role_map[x])
if not role or not isinstance(role, str):
    role = "general_physician"

# -------------------------
# Home Page
# -------------------------


if page == "🏞️ Home":
    t = TRANSLATIONS[output_lang]
    st.title("🩺 " + t["page1_title"])
    st.header(t["page1_header"])
    st.markdown(t["page1_subheader"])

# -------------------------
# Image Analysis with Chat
# -------------------------


if page == "🖼️ Image Analysis":
    t = TRANSLATIONS[output_lang]
    st.subheader("🖼️ " + t["image_upload"])
    image_file = st.file_uploader(t["image_upload"], type=["png", "jpg", "jpeg"])

    # Gemini AI Image Analysis Block
    st.markdown(f"### {t.get('gemini_image_analysis_title', 'AI Image Analysis (Gemini)')}")
    st.markdown(t.get('gemini_image_analysis_disclaimer', "**For Educational Use Only.** This tool uses a general-purpose AI and is not a certified medical device. Do not use it for self-diagnosis. Always consult a real doctor."))

    # --- Gemini Imports and Setup ---
    import google.generativeai as genai
    import io
    from dotenv import load_dotenv
    load_dotenv()
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        st.error(f"{t.get('gemini_image_analysis_error_config', 'Error configuring Google API:')} GOOGLE_API_KEY not found.")
        st.error(t.get('gemini_image_analysis_error_key', 'Please make sure you have a GOOGLE_API_KEY in your .env file in the root directory.'))
    else:
        try:
            genai.configure(api_key=gemini_api_key)
        except Exception as e:
            st.error(f"{t.get('gemini_image_analysis_error_config', 'Error configuring Google API:')} {e}")
            st.stop()

    PROMPT_TEMPLATE = """
You are a highly skilled medical expert with specializations across multiple fields like radiology, ophthalmology, and dermatology. Your task is to analyze the provided image.

**Instructions:**
1.  **Identify the Image Type:** First, determine the type of medical image (e.g., X-ray of a bone, clinical photograph of an eye, dermatological image of a skin condition, etc.).
2.  **Adopt the Correct Persona:** Based on the image type, adopt the appropriate expert role.
    *   For an X-ray, act as a **Radiologist**.
    *   For an eye image, act as an **Ophthalmologist**.
    *   For a skin image, act as a **Dermatologist**.
    *   For other images, use your best judgment to select a relevant medical expert role.
3.  **Provide a Structured Report:** Generate a detailed analysis in a structured format using markdown. The report should include the following sections:
    *   `### Role Adopted:` (State the expert role you have taken on).
    *   `### Observations:` (Describe what you see in the image in medical terms).
    *   `### Impression / Potential Diagnosis:` (Provide a potential diagnosis or impression based on the visual evidence).
    *   `### Recommendations:` (Suggest potential next steps, such as specific tests or consultation with a specialist).
4.  **Crucial Disclaimer:** Conclude your analysis with the following mandatory disclaimer, formatted exactly as shown:

---
***Disclaimer:*** This is an AI-generated analysis for educational and informational purposes only. It is **NOT** a substitute for a professional medical diagnosis. Please consult a qualified healthcare provider for any health concerns.*
"""

    if image_file is not None:
        image_bytes = image_file.read()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(t.get('gemini_image_analysis_uploaded_image', 'Uploaded Image'))
            st.image(image_bytes, caption=t.get('gemini_image_analysis_uploaded_image', 'Image ready for analysis'), use_column_width=True)
        with col2:
            st.subheader(t.get('gemini_image_analysis_ai_analysis', 'AI Analysis'))
            if st.button(t.get('gemini_image_analysis_button', 'Analyze Image with Gemini AI'), type="primary") or ("image_analysis_done" not in st.session_state):
                with st.spinner(t.get('gemini_image_analysis_spinner', 'The AI expert is examining the image... Please wait.')):
                    try:
                        from PIL import Image as PILImage
                        image_part = PILImage.open(io.BytesIO(image_bytes))
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content([PROMPT_TEMPLATE, image_part])
                        analysis_text = response.text
                    except Exception as e:
                        analysis_text = f"{t.get('gemini_image_analysis_error', 'An error occurred during AI analysis:')} {e}"
                    st.session_state.image_analysis = analysis_text
                    st.session_state.image_analysis_displayed = False
                    st.session_state.image_messages = []
                    st.session_state["image_analysis_done"] = True
            if st.session_state.get("image_analysis_done"):
                st.markdown(st.session_state.image_analysis)
                if st.button(t.get('gemini_image_analysis_voice', '🔊 Listen to AI Analysis'), key="gemini_voice"):
                    st.session_state["gemini_image_analysis_spoken"] = False
                if not st.session_state.get("gemini_image_analysis_spoken", False):
                    st.components.v1.html(generate_audio_html(st.session_state.image_analysis, lang=output_lang, key="gemini_image_analysis"), height=100)
                    st.session_state["gemini_image_analysis_spoken"] = True



    if st.session_state.image_analysis:
        st.divider()
        st.subheader("💬 " + t["ask_image_question"])

        if not st.session_state.image_analysis_displayed:
            with st.chat_message("assistant"):
                st.markdown(st.session_state.image_analysis)
                st.session_state.image_analysis_spoken = False
                if st.button("🔊 Resume Audio", key="resume_image_analysis"):
                    st.session_state.image_analysis_spoken = False
                if not st.session_state.image_analysis_spoken:
                    st.components.v1.html(generate_audio_html(st.session_state.image_analysis, lang=output_lang, key="image_analysis"), height=100)
                    st.session_state.image_analysis_spoken = True
            st.session_state.image_analysis_displayed = True

        user_input = st.chat_input(t["ask_image_question"], key="image_chat")

        if user_input:
            st.session_state.image_messages.append(("user", user_input))
            with st.spinner("Thinking..."):
                history = [f"Image Analysis: {st.session_state.image_analysis}"]
                for role_msg, msg in st.session_state.image_messages:
                    history.append(f"{role_msg.capitalize()}: {msg}")
                # Ensure role is always a string and fallback to 'general_physician' if missing
                role_value = str(role) if role else "general_physician"
                if not role_value or role_value == "None":
                    role_value = "general_physician"
                payload = {
                    "question": user_input,
                    "context": "\n".join(history),
                    "tone": tone,
                    "role": role_value,
                    "simplify": simple_explanation
                }
                res = requests.post(f"{BACKEND_URL}/ask", data=payload)
                try:
                    answer = res.json().get("answer", "❌ No response")
                except:
                    answer = "❌ Invalid response from backend."
                translated_answer = translate_answer(answer, output_lang)
                st.session_state.image_messages.append(("assistant", translated_answer))

        for idx, (role, msg) in enumerate(st.session_state.image_messages):
            with st.chat_message(role):
                st.markdown(msg)
                if role == "assistant":
                    if st.button(f"🔊 Resume Audio {idx+1}", key=f"resume_image_{idx}"):
                        st.session_state[f"assistant_image_spoken_{idx}"] = False
                    if not st.session_state.get(f"assistant_image_spoken_{idx}", False):
                        st.components.v1.html(generate_audio_html(msg, lang=output_lang, key=f"assistant_image_{idx}"), height=100)
                        st.session_state[f"assistant_image_spoken_{idx}"] = True

# -------------------------
# PDF Report Analysis with Chat
# -------------------------


if page == "📄 PDF Report Analysis":
    t_pdf = TRANSLATIONS.get(output_lang, TRANSLATIONS["en"])
    st.subheader("📄 " + t_pdf.get("pdf_upload", TRANSLATIONS["en"]["pdf_upload"]))
    pdf_file = st.file_uploader(t_pdf.get("pdf_upload", TRANSLATIONS["en"]["pdf_upload"]), type=["pdf"])

    if pdf_file:
        with st.spinner(t_pdf.get("pdf_analysis", TRANSLATIONS["en"]["pdf_analysis"]) + "..."):
            pdf_file.seek(0)
            try:
                res = requests.post(f"{BACKEND_URL}/upload_pdf", files={"file": pdf_file})
                if res.status_code != 200:
                    st.error(f"❌ {t_pdf.get('backend_error', TRANSLATIONS['en']['backend_error'])}: {res.status_code} - {res.text}")
                    result = None
                else:
                    res_json = res.json()
                    result = res_json.get("content", "❌ Failed to analyze PDF")
            except Exception as e:
                st.error(f"❌ {t_pdf.get('analysis_error', TRANSLATIONS['en']['analysis_error'])}: {e}")
                result = None

            if result:
                translated_result = translate_answer(result, output_lang)
                st.session_state.pdf_analysis = translated_result
                st.session_state.pdf_analysis_displayed = False
                st.session_state.pdf_messages = []

    if st.session_state.pdf_analysis:
        st.divider()
        st.subheader("📁 " + t_pdf.get("ask_pdf_question", TRANSLATIONS["en"]["ask_pdf_question"]))

        if not st.session_state.pdf_analysis_displayed:
            with st.chat_message("assistant"):
                st.markdown(st.session_state.pdf_analysis)
                st.session_state.pdf_analysis_spoken = False
                if st.button("🔊 Resume Audio", key="resume_pdf_analysis"):
                    st.session_state.pdf_analysis_spoken = False
                if not st.session_state.pdf_analysis_spoken:
                    st.components.v1.html(generate_audio_html(st.session_state.pdf_analysis, lang=output_lang, key="pdf_analysis"), height=100)
                    st.session_state.pdf_analysis_spoken = True
            st.session_state.pdf_analysis_displayed = True

        user_input = st.chat_input(t_pdf.get("ask_pdf_question", TRANSLATIONS["en"]["ask_pdf_question"]), key="pdf_chat")

        if user_input:
            st.session_state.pdf_messages.append(("user", user_input))
            with st.spinner("Thinking..."):
                history = [f"PDF Analysis: {st.session_state.pdf_analysis}"]
                for role_msg, msg in st.session_state.pdf_messages:
                    history.append(f"{role_msg.capitalize()}: {msg}")
                payload = {
                    "question": user_input,
                    "context": "\n".join(history),
                    "tone": tone,
                    "role": str(role) if role else "general_physician",
                    "simplify": simple_explanation
                }
                res = requests.post(f"{BACKEND_URL}/ask", data=payload)
                try:
                    answer = res.json().get("answer", "❌ No response")
                except:
                    answer = "❌ Invalid response from backend."
                translated_answer = translate_answer(answer, output_lang)
                st.session_state.pdf_messages.append(("assistant", translated_answer))

        for idx, (role, msg) in enumerate(st.session_state.pdf_messages):
            with st.chat_message(role):
                st.markdown(msg)
                if role == "assistant":
                    if st.button(f"🔊 Resume Audio {idx+1}", key=f"resume_pdf_{idx}"):
                        st.session_state[f"assistant_pdf_spoken_{idx}"] = False
                    if not st.session_state.get(f"assistant_pdf_spoken_{idx}", False):
                        st.components.v1.html(generate_audio_html(msg, lang=output_lang, key=f"assistant_pdf_{idx}"), height=100)
                        st.session_state[f"assistant_pdf_spoken_{idx}"] = True


# -------------------------
# 🎙️ Voice Q&A Page
# -------------------------


if page == "🎙️ Voice Q&A":
    # Add a dedicated language selector for Voice Q&A UI
    voice_ui_lang = st.selectbox(
        "🌐 Voice Q&A UI language:",
        list(language_map.keys()),
        index=list(language_map.keys()).index(output_lang),
        format_func=lambda x: language_map[x],
        key="voice_ui_lang_selectbox"
    )
    t_voice = TRANSLATIONS.get(voice_ui_lang, TRANSLATIONS["en"])
    st.title("🎙️ " + t_voice.get("voice_title", TRANSLATIONS["en"].get("voice_title", "Voice Q&A")))
    st.markdown(t_voice.get("voice_instruction", TRANSLATIONS["en"].get("voice_instruction", "Speak your question below.")))

    # Language code map for Google API
    lang_code_map = {
        "en": "en-US", "fr": "fr-FR", "es": "es-ES",
        "de": "de-DE", "hi": "hi-IN", "zh": "zh-CN"
    }

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Voice input box
    st.write("🎤 " + t_voice.get("voice_input", TRANSLATIONS["en"]["voice_input"]))
    spoken_text = record_and_transcribe(lang=lang_code_map.get(input_lang, "en-US"))

    if spoken_text and not spoken_text.startswith("❌"):
        st.session_state.messages.append(("user", spoken_text))
        st.markdown("**" + t_voice.get("voice_response", TRANSLATIONS["en"]["voice_response"]) + "**")
        st.markdown(f"> {spoken_text}")

        # Translate to English if needed
        translated_question = translate_question(spoken_text, input_lang)

        # Prepare payload
        payload = {
            "question": translated_question,
            "context": "",
            "tone": tone,
            "role": str(role),
            "simplify": simple_explanation
        }

        # Get answer from backend (no timeout limit)
        with st.spinner("Thinking..."):
            try:
                res = requests.post(f"{BACKEND_URL}/ask", data=payload)
                answer = res.json().get("answer", "❌ No response")
            except requests.exceptions.ConnectionError:
                answer = "❌ Backend connection error: Unable to reach the server. Please check if the backend is running."
            except Exception as e:
                answer = f"❌ Invalid response from backend: {e}"

        translated_answer = translate_answer(answer, output_lang)
        st.session_state.messages.append(("assistant", translated_answer))

    elif spoken_text.startswith("❌"):
        st.warning(spoken_text)

    # Chat-style rendering of messages
    for idx, (role_label, msg) in enumerate(st.session_state.messages):
        with st.chat_message(role_label):
            st.markdown(msg)
            if role_label == "assistant":
                st.components.v1.html(
                    generate_audio_html(msg, lang=output_lang, key=f"voice_qna_{idx}"),
                    height=100
                )

    # Re-record section at the bottom
    st.divider()
    st.markdown("🔁 " + t_voice.get("ask_another", TRANSLATIONS["en"]["ask_another"]))


# -------------------------
# 💬 Chatbot Q&A
# -------------------------


if page == "💬 Chatbot Q&A":
    t = TRANSLATIONS.get(output_lang, TRANSLATIONS["en"])
    st.title("💬 " + t.get("chatbot_title", TRANSLATIONS["en"]["chatbot_title"]))
    user_input = st.chat_input(t.get("chatbot_input", TRANSLATIONS["en"]["chatbot_input"]))

    if user_input:
        st.session_state.messages.append(("user", user_input))
        user_msg = user_input

        if is_greeting(user_msg):
            st.session_state.messages.append(("assistant", t.get("greeting", TRANSLATIONS["en"]["greeting"])))
        else:
            with st.spinner("Thinking..."):
                history = []
                for role_msg, msg in st.session_state.messages:
                    if role_msg == "user":
                        history.append(f"User: {msg}")
                    elif role_msg == "assistant":
                        history.append(f"Assistant: {msg}")
                if st.session_state.image_analysis:
                    history.append(f"Image Analysis: {st.session_state.image_analysis}")
                if st.session_state.pdf_analysis:
                    history.append(f"PDF Analysis: {st.session_state.pdf_analysis}")
                chat_history = "\n".join(history)
                payload = {
                    "question": user_msg,
                    "context": chat_history,
                    "tone": tone,
                    "role": str(role) if role else "general_physician",
                    "simplify": True
                }
                try:
                    res = requests.post(f"{BACKEND_URL}/ask", data=payload)
                    answer = res.json().get("answer", "❌ No response")
                except requests.exceptions.ConnectionError:
                    answer = "❌ Backend connection error: Unable to reach the server. Please check if the backend is running."
                except Exception as e:
                    answer = f"❌ Unexpected error: {e}"
                translated_answer = translate_answer(answer, output_lang)
                st.session_state.messages.append(("assistant", translated_answer))

    for idx, (role_label, msg) in enumerate(st.session_state.messages):
        with st.chat_message(role_label):
            st.markdown(msg)
            if role_label == "assistant":
                if st.button(f"🔊 Resume Audio {idx+1}", key=f"resume_chatbot_{idx}"):
                    st.session_state[f"assistant_chatbot_spoken_{idx}"] = False
                if not st.session_state.get(f"assistant_chatbot_spoken_{idx}", False):
                    speak_text(msg, key=f"assistant_{idx}")
                    st.session_state[f"assistant_chatbot_spoken_{idx}"] = True


# -------------------------
# 🩹 Symptom Bot
# -------------------------


if page == "🩹 Symptom Bot":
    t = TRANSLATIONS[output_lang]
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        st.error("Gemini API key not found. Please set GEMINI_API_KEY in .env file.")
        st.stop()

    # Configure Gemini
    genai.configure(api_key=GEMINI_API_KEY)

    # Load Gemini model
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        st.error(f"Error loading Gemini model: {e}")
        st.stop()

    # Streamlit UI
    st.title("🩺 " + t["symptom_bot_title"])
    st.subheader("📝 " + t["symptom_bot_subheader"])

    col1, col2 = st.columns(2)
    with col1:
        gender = st.radio(t["gender"], [t["male"], t["female"]])
    with col2:
        age = st.number_input(t["age"], min_value=0, step=1)
    if gender == t["female"]:
        pregnant = st.radio(t["pregnant"], [t["no"], t["yes"]], horizontal=True)
    else:
        pregnant = t["no"]

    history = st.text_area(t["history"], placeholder=t["hist_example"])
    symptoms = st.text_area(t["symptoms"], placeholder=t["symp_example"])
    exam_findings = st.text_area(t["exam"], placeholder=t["exam_example"])
    lab_results = st.text_area(t["lab"], placeholder=t["lab_example"])

    # Submit button
    if st.button(t["submit"]):
        with st.spinner(t["submit_wait"]):
            prompt = f"""
You are a medical diagnostic assistant. Given the patient data, provide:
1. A summarized medical report.
2. A list of possible differential diagnoses with short reasoning.

Patient Info:
- Gender: {gender}
- Age: {age}
- Pregnant: {pregnant}

Medical History: {history}
Symptoms: {symptoms}
Examination Findings: {exam_findings}
Lab Results: {lab_results}
"""
            try:
                response = model.generate_content(prompt)
                result = response.text
            except Exception as e:
                result = f"❌ Error communicating with Gemini: {e}"

        # Output
        st.subheader("📄 " + t.get("summary", "Summary"))
        st.markdown(f"""
**{t.get('vissum_patient', 'Patient: ')}** {gender}, {age}{t.get('vissum_yrsold', ' yrs old')}  
**{t.get('vissum_pregnancy', 'Pregnancy: ')}** {pregnant}  
**{t.get('vissum_history', 'History: ')}** {history or t.get('none', 'none')}  
**{t.get('vissum_symp', 'Symptoms: ')}** {symptoms or t.get('none', 'none')}  
**{t.get('vissum_exam', 'Exam findings: ')}** {exam_findings or t.get('none', 'none')}  
**{t.get('vissum_lab', 'Lab results: ')}** {lab_results or t.get('none', 'none')}  
""")

        st.subheader("🧠 " + t["diagnostic"])
        st.markdown(result)