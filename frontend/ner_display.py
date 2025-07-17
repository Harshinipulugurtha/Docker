# frontend/ner_display.py
import streamlit as st
import re

def display_ner_highlighted(text):
    styles = {
        "🦠": "background-color:#fdd;",
        "🤒": "background-color:#dff;",
        "💊": "background-color:#ffd;"
    }

    for emoji, style in styles.items():
        pattern = rf"\*\*{emoji} (.*?)\*\*"
        text = re.sub(pattern, rf"<span style='{style}'>{emoji} \1</span>", text)

    st.markdown(text, unsafe_allow_html=True)
