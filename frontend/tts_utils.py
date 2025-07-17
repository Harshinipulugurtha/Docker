# frontend/tts_utils.py
from gtts import gTTS
from io import BytesIO
import base64
import streamlit as st
from bs4 import BeautifulSoup

def clean_text(text):
    try:
        # Remove colons and strip HTML tags
        text = text.replace(':', '')
        from bs4 import BeautifulSoup
        return BeautifulSoup(text, "html.parser").get_text().strip()[:3000]
    except:
        return text.replace(':', '')[:3000]

def speak_text(text, lang="en", key=None):
    try:
        cleaned = clean_text(text)
        tts = gTTS(cleaned, lang=lang)
        with BytesIO() as audio_file:
            tts.write_to_fp(audio_file)
            audio_file.seek(0)
            b64 = base64.b64encode(audio_file.read()).decode()
        audio_html = f"""
        <div>
            <audio id="audio{key}" autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            <div style="margin-top: 10px;">
                <button onclick="document.getElementById('audio{key}').pause()">⏸ Pause</button>
                <button onclick="document.getElementById('audio{key}').play()">▶ Resume</button>
            </div>
        </div>
        """
        st.components.v1.html(audio_html, height=100)
    except Exception as e:
        st.error(f"TTS failed: {e}")

def generate_audio_html(text, lang="en", key=None):
    try:
        cleaned = clean_text(text)
        tts = gTTS(cleaned, lang=lang)
        with BytesIO() as audio_file:
            tts.write_to_fp(audio_file)
            audio_file.seek(0)
            b64 = base64.b64encode(audio_file.read()).decode()
        return f"""
        <div>
            <audio id="audio{key}" autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            <div style="margin-top: 10px;">
                <button onclick="document.getElementById('audio{key}').pause()">⏸ Pause</button>
                <button onclick="document.getElementById('audio{key}').play()">▶ Resume</button>
            </div>
        </div>
        """
    except Exception as e:
        return f"<p style='color:red;'>TTS failed: {e}</p>"
