import streamlit as st
from streamlit.components.v1 import html
st.set_page_config(
    page_title="Projekt S.A.N.A",
    page_icon="https://avatars.githubusercontent.com/u/175069629?v=4",
    layout="wide"
)

import wikipedia
import wolframalpha
import google.generativeai as genai
import logging
import base64
import requests
import numpy as np
from PIL import Image, ImageFilter
import io

# For OCR
import pytesseract

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

logo = "https://avatars.githubusercontent.com/u/175069629?v=4"

#####################################
#         API Configuration         #
#####################################

try:
    GENAI_API_KEY = st.secrets["GENAI_API_KEY"]
    genai.configure(api_key=GENAI_API_KEY)
    system_prompt = "You are S.A.N.A (Secure Autonomous Non-Intrusive Assistant), a smart, privacy-respecting AI"
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        system_instruction=[system_prompt]
    )
except KeyError:
    st.error("Error: GENAI_API_KEY not found in Streamlit secrets.")
    model = None

try:
    APP_ID = st.secrets["APP_ID"]
    wolfram_client = wolframalpha.Client(APP_ID)
except KeyError:
    st.error("Error: APP_ID not found in Streamlit secrets.")
    wolfram_client = None

HF_API_KEY = st.secrets["HF_API_KEY"]
HF_IMAGE_MODEL = "Salesforce/blip-image-captioning-large"
HF_SUMMARY_MODEL = "facebook/bart-large-cnn"
HF_GEN_MODEL = "stabilityai/stable-diffusion-2"

#####################################
#          Core Functions           #
#####################################

def search_wikipedia(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return "Multiple meanings detected: " + ", ".join(e.options[:5])
    except wikipedia.exceptions.PageError:
        return "No results found."
    except Exception as e:
        logging.error(f"Wikipedia error: {e}")
        return "Error while searching Wikipedia."

def query_wolfram_alpha(query):
    if wolfram_client is None:
        return "Wolfram Alpha not configured."
    try:
        res = wolfram_client.query(query)
        return next(res.results).text
    except Exception as e:
        logging.error(f"Wolfram Alpha error: {e}")
        return "Error querying Wolfram Alpha."

def query_google_gemini(query, context):
    if model is None:
        return "Gemini is not configured."
    try:
        response = model.generate_content(context + f"\nUser: {query}\nAssistant:")
        return response.text
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "Error fetching from Gemini."

def summarize_text_with_gemini(text):
    if model is None:
        return "Gemini is not configured."
    try:
        response = model.generate_content(f"Summarize the following text:\n\n{text}")
        return response.text
    except Exception as e:
        logging.error(f"Gemini Summary error: {e}")
        return "Error summarizing the text."

def process_uploaded_file(uploaded_file):
    try:
        if uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        else:
            return "Unsupported file type."
        return summarize_text_with_gemini(text)
    except Exception as e:
        logging.error(f"File processing error: {e}")
        return "Error processing the file."

def describe_image(image):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    payload = {"inputs": encoded_image}
    
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}",
            headers=headers,
            json=payload
        )
        return response.json()[0]['generated_text']
    except Exception as e:
        logging.error(f"Image description error: {e}")
        return "Error describing the image."

def generate_image(prompt):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    data = {"inputs": prompt}

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
            headers=headers,
            json=data
        )

        if "image" in response.headers.get("Content-Type", ""):
            image = Image.open(io.BytesIO(response.content))
            return image
        else:
            error_message = response.json()
            logging.error(f"Image generation error: {error_message}")
            return None
    except Exception as e:
        logging.error(f"Image generation error: {e}")
        return None

#####################################
#     Enhanced Image Processing     #
#####################################

# 1. Image OCR using pytesseract
def image_ocr(image):
    image = image.convert("RGB")
    return pytesseract.image_to_string(image)

# 2. Image Filtering using PIL filters
def apply_filter(image, filter_type="BLUR"):
    if filter_type == "BLUR":
        return image.filter(ImageFilter.BLUR)
    elif filter_type == "CONTOUR":
        return image.filter(ImageFilter.CONTOUR)
    elif filter_type == "DETAIL":
        return image.filter(ImageFilter.DETAIL)
    else:
        return image

#####################################
#          Streamlit UI             #
#####################################

st.markdown(
    f"""
    <div style='display: flex; align-items: center;'>
        <img src='{logo}' width='50' style='margin-right: 15px;'>
        <h1 style='margin: 0;'>Projekt S.A.N.A</h1>
    </div>
    """, unsafe_allow_html=True
)

st.markdown("**S**ecure **A**utonomous and **N**on-Intrusive **A**ssistant")

with st.sidebar:
    st.title("S.A.N.A Settings")
    st.markdown("⚙️ **Customize your assistant experience (coming soon!)**")
    st.markdown("---")
    feature = st.selectbox("Select a feature:", [
        "General Chat", 
        "Wikipedia Search", 
        "Wolfram Alpha Queries", 
        "PDF/TXT Summary", 
        "Image Description", 
        "Image Generation",
        "Image OCR",
        "Image Filtering"
    ])
    
    # Clear History Button
    if st.button("Clear Chat History"):
        st.session_state["chat_history"] = []
        st.session_state["context"] = ""
        st.session_state["pdf_summary_done"] = False
        st.rerun()

    # Display old chats in the sidebar
    if "old_chats" not in st.session_state:
        st.session_state["old_chats"] = []
    
    with st.expander("Old Chats"):
        for i, chat in enumerate(st.session_state["old_chats"]):
            if st.button(f"Load Chat {i+1}"):
                st.session_state["chat_history"] = chat["history"]
                st.session_state["context"] = chat["context"]
                st.rerun()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "context" not in st.session_state:
    st.session_state["context"] = ""

###############################
#      Chat History & Chat    #
###############################

st.header("💬 Chat History")
st.write("---")
for sender, message in st.session_state["chat_history"]:
    if sender == "You":
        with st.chat_message("user"):
            st.write(message)
    else:
        with st.chat_message("assistant"):
            st.write(message)
st.write("---")

# Display the input field for relevant features
if feature in ["General Chat", "Wikipedia Search", "Wolfram Alpha Queries"]:
    with st.form("InputForm"):
        user_input = st.text_input("💬 Type your query:", placeholder="Ask anything...", key="user_input", autocomplete="off")
        if st.form_submit_button("Send"):
            if user_input:
                st.session_state["chat_history"].append(("You", user_input))
                if feature == "Wikipedia Search":
                    response = search_wikipedia(user_input)
                elif feature == "Wolfram Alpha Queries":
                    response = query_wolfram_alpha(user_input)
                elif feature == "General Chat":
                    response = query_google_gemini(user_input, st.session_state["context"])
                else:
                    response = "Invalid feature."
                st.session_state["chat_history"].append(("S.A.N.A", response))
                st.session_state["context"] += f"User: {user_input}\nAssistant: {response}\n"
                st.rerun()

# Save chat history
if st.button("Save Chat"):
    st.session_state["old_chats"].append({
        "history": st.session_state["chat_history"],
        "context": st.session_state["context"]
    })
    st.success("Chat history saved!")
