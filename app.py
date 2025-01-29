import streamlit as st
import wikipedia
import wolframalpha
import google.generativeai as genai
import logging
import base64
import requests
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# APP logo
logo = "https://avatars.githubusercontent.com/u/175069629?v=4"

# Google Gemini API key
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

# WolframAlpha API key
try:
    APP_ID = st.secrets["APP_ID"]
    wolfram_client = wolframalpha.Client(APP_ID)
except KeyError:
    st.error("Error: APP_ID not found in Streamlit secrets.")
    wolfram_client = None

# Hugging Face API for Image Description & PDF Summary
HF_API_KEY = st.secrets["HF_API_KEY"]
HF_IMAGE_MODEL = "Salesforce/blip-image-captioning-large"
HF_SUMMARY_MODEL = "facebook/bart-large-cnn"

# Function: Wikipedia Search
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

# Function: Wolfram Alpha Query
def query_wolfram_alpha(query):
    if wolfram_client is None:
        return "Wolfram Alpha not configured."
    try:
        res = wolfram_client.query(query)
        return next(res.results).text
    except Exception as e:
        logging.error(f"Wolfram Alpha error: {e}")
        return "Error querying Wolfram Alpha."

# Function: Gemini Chat
def query_google_gemini(query, context):
    if model is None:
        return "Gemini is not configured."
    try:
        response = model.generate_content(context + f"\nUser: {query}\nAssistant:")
        return response.text
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "Error fetching from Gemini."

# Function: PDF/TXT Summarization using Hugging Face
def summarize_text(text):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    data = {"inputs": text, "parameters": {"max_length": 150, "min_length": 50, "do_sample": False}}
    try:
        response = requests.post(f"https://api-inference.huggingface.co/models/{HF_SUMMARY_MODEL}", headers=headers, json=data)
        return response.json()[0]['summary_text']
    except Exception as e:
        logging.error(f"PDF/TXT Summary error: {e}")
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
        return summarize_text(text)
    except Exception as e:
        logging.error(f"File processing error: {e}")
        return "Error processing the file."

# Function: Image Description using Hugging Face
def describe_image(image):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    payload = {"inputs": encoded_image}
    
    try:
        response = requests.post(f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}", headers=headers, json=payload)
        return response.json()[0]['generated_text']
    except Exception as e:
        logging.error(f"Image description error: {e}")
        return "Error describing the image."

# Streamlit App
st.set_page_config(page_title="Projekt S.A.N.A", page_icon=logo, layout="wide")

# Title with Logo
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="{logo}" width="50" style="margin-right: 15px;">
        <h1 style="margin: 0;">Projekt S.A.N.A</h1>
    </div>
    """, unsafe_allow_html=True
)

st.markdown("**S.A.N.A** is a secure, autonomous, and non-intrusive virtual assistant. 😊")

# Sidebar
with st.sidebar:
    st.title("S.A.N.A Settings")
    st.markdown("⚙️ **Customize your assistant experience (coming soon!)**")
    st.markdown("---")
    st.markdown("1. Wikipedia Search\n2. Wolfram Alpha Queries\n3. Google Gemini Chat\n4. PDF/TXT Summary\n5. Image Description")

# Initialize session variables
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "context" not in st.session_state:
    st.session_state["context"] = ""

# Feature Selection
feature = st.selectbox("Select a feature:", ["General Chat", "Wikipedia Search", "Wolfram Alpha Queries", "PDF/TXT Summary", "Image Description"])

# Display Chat History
st.markdown("### 💬 Chat History")
st.write("---")
for sender, message in st.session_state["chat_history"]:
    if sender == "You":
        st.markdown(f"**🧑‍💻 You:** {message}")
    else:
        st.markdown(f"<b>S.A.N.A:</b> {message}", unsafe_allow_html=True)

st.write("---")

# User Input
user_input = st.text_input("💬 Type your query:", placeholder="Ask anything...", key="user_input")

if st.button("Send"):
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
        st.experimental_rerun()

# PDF/TXT Summary
if feature == "PDF/TXT Summary":
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file:
        st.success("File uploaded successfully!")
        summary = process_uploaded_file(uploaded_file)
        st.markdown(f"**📜 Summary:** {summary}")

# Image Description
if feature == "Image Description":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        description = describe_image(image)
        st.markdown(f"**🖼️ Description:** {description}")

    # Take Picture Button
    if st.camera_input("Take a picture"):
        captured_image = Image.open(st.camera_input("Take a picture"))
        st.image(captured_image, caption="Captured Image", use_column_width=True)
        description = describe_image(captured_image)
        st.markdown(f"**🖼️ Description:** {description}")
