import streamlit as st
from PIL import Image
import google.generativeai as genai
import os
import io
from dotenv import load_dotenv
import fitz  # PyMuPDF <<< NEW IMPORT

# --- Load Environment Variables ---
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Medical Report Analyzer",
    page_icon="🩺",
    layout="wide"
)

# --- Google Gemini API Configuration ---
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please check your .env file.")
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Error configuring Google API: {e}")
    st.error("Please ensure you have a GOOGLE_API_KEY in a .env file.")
    st.stop()


# --- <<< NEW FUNCTION to handle both PDF and Image files >>> ---
def process_uploaded_file(uploaded_file):
    """
    Converts an uploaded file (image or PDF) into a list of PIL Image objects.
    """
    if uploaded_file.type == "application/pdf":
        images = []
        try:
            # Open the PDF file from bytes
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                # Render page to a pixmap (an image)
                pix = page.get_pixmap(dpi=300) # Higher DPI for better quality
                # Convert pixmap to a PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            pdf_document.close()
            return images
        except Exception as e:
            st.error(f"Error processing PDF file: {e}")
            return None
    else:
        # It's an image file
        try:
            return [Image.open(uploaded_file)]
        except Exception as e:
            st.error(f"Error processing image file: {e}")
            return None

# --- <<< MODIFIED FUNCTION to handle a list of images >>> ---
def get_gemini_analysis(list_of_pil_images, prompt):
    """
    Sends a list of images and a prompt to the Gemini model.
    """
    if not list_of_pil_images:
        return "Could not process the uploaded file. Please try again."
        
    try:
        # Using Gemini 1.5 Flash
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # Prepare the content list. Start with the prompt.
        content = [prompt]
        # Add all the images to the content list
        content.extend(list_of_pil_images)
        
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        return f"An error occurred during AI analysis: {e}"


# --- The Master Prompt for Lab Report Analysis (Unchanged) ---
PROMPT_TEMPLATE = """
You are an expert physician, either a General Practitioner or an Endocrinologist, skilled at interpreting laboratory results and explaining them to patients in a clear, understandable manner. Your task is to analyze the provided medical lab report images.

**Instructions:**
1.  **Extract Key Data:** Carefully read the entire report across all provided pages. Identify key biomarkers, their results, units, and reference ranges. Pay close attention to any values flagged as 'high' or 'low'.
2.  **Summarize Findings:** Provide a structured report using markdown. The report must include the following sections:

    ### 🩺 Lab Report Summary for Patient
    
    ### **Key Findings at a Glance**
    *   Create a simple bulleted list of the most significant results that are outside the normal reference range. For each, state the biomarker, the result, and whether it's High or Low.
    
    ### **Detailed Breakdown**
    *   **Glucose:** Analyze the fasting glucose level. Explain what it means in the context of diabetes risk (normal, pre-diabetes, etc.).
    *   **Lipid Panel (Cholesterol, etc.):**
        *   **Total Cholesterol:** Explain the result (e.g., borderline, high).
        *   **Triglycerides:** Explain this result. Note if it's significantly elevated as this is a major health indicator.
        *   **HDL Cholesterol ('Good' Cholesterol):** Explain the result, noting if it's low.
        *   **LDL Cholesterol ('Bad' Cholesterol):** Explain this result.
    *   **HbA1c (if present):** Explain what HbA1c represents (average blood sugar over 2-3 months) and interpret the result.
    *   **Other Notable Results (if any):** Briefly comment on any other out-of-range results like Urea, Creatinine, etc.

    ### **Overall Impression & Recommendations**
    *   Provide a concluding paragraph that synthesizes the findings. For example, "Overall, this report shows a normal fasting glucose level but indicates dyslipidemia (abnormal cholesterol and triglyceride levels)..."
    *   Provide general, non-prescriptive recommendations, such as "It is highly recommended to discuss these results with your physician, who may suggest lifestyle changes (diet, exercise) and potentially medication to manage your cholesterol and triglyceride levels."

3.  **Crucial Disclaimer:** Conclude your entire analysis with the following mandatory disclaimer:

---
***Disclaimer:** This is an AI-generated analysis for educational purposes. It is **NOT** a substitute for a professional medical diagnosis or consultation. Please consult your qualified healthcare provider to discuss these results and determine the appropriate next steps for your health.*
"""

# --- Streamlit App UI ---
st.title("🩺 AI Medical Report Analyzer")
st.markdown("Upload a lab report (Image or PDF), and the AI will provide a simplified explanation.")

st.warning(
    "**For Educational Use Only.** This tool is not a certified medical device and should not be used for self-diagnosis. "
    "Always consult a real doctor to interpret your results."
)

# --- <<< MODIFIED Uploader to accept PDF >>> ---
uploaded_file = st.file_uploader(
    "Upload a lab report...",
    type=["jpg", "jpeg", "png", "pdf"]
)

if uploaded_file is not None:
    # --- <<< MODIFIED Logic to use the new processing function >>> ---
    pil_images = process_uploaded_file(uploaded_file)

    if pil_images:
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.subheader("Uploaded Report Preview")
            # Show a preview of the first page
            st.image(pil_images[0], caption=f"Preview (Page 1 of {len(pil_images)})", use_column_width=True)

        with col2:
            st.subheader("AI-Powered Explanation")
            if st.button("Analyze Lab Report", type="primary"):
                with st.spinner('The AI physician is reviewing the report... Please wait.'):
                    # Call the AI function with the list of processed images
                    analysis_text = get_gemini_analysis(pil_images, PROMPT_TEMPLATE)
                    st.markdown(analysis_text)