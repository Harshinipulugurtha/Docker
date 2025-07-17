# frontend/mic_utils.py

import speech_recognition as sr
from pydub import AudioSegment
import tempfile
from streamlit_mic_recorder import mic_recorder
from io import BytesIO

def record_and_transcribe(lang="en-US"):
    audio = mic_recorder(
        start_prompt="🎙️ Speak", 
        stop_prompt="⏹️ Stop", 
        just_once=True, 
        key="mic"
    )

    if not audio or "bytes" not in audio:
        return ""

    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        sound = AudioSegment.from_file(BytesIO(audio["bytes"]))
        sound.export(f.name, format="wav")
        with sr.AudioFile(f.name) as source:
            audio_data = recognizer.record(source)
            try:
                return recognizer.recognize_google(audio_data, language=lang)
            except sr.UnknownValueError:
                return "❌ Could not understand the audio. Please try again."
            except sr.RequestError as e:
                return f"❌ Could not reach the speech service: {e}"
            except Exception as e:
                return f"❌ Audio transcription error: {e}"
