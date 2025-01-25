import streamlit as st
import wikipedia
import wolframalpha
import google.generativeai as genai
import logging
from PyPDF2 import PdfReader
import os

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Google Gemini API key and model initialization
try:
    GENAI_API_KEY = st.secrets["GENAI_API_KEY"]
    genai.configure(api_key=GENAI_API_KEY)
    system_prompt = '''You are S.A.N.A (Secure Autonomous Non-Intrusive Assistant), a smart, privacy-respecting AI'''
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        system_instruction=[system_prompt]
    )
except KeyError:
    st.error("Error: GENAI_API_KEY not found in Streamlit secrets. Please configure it.")
    model = None
except Exception as e:
    st.error(f"Error initializing Gemini: {e}")
    model = None

# WolframAlpha App ID and client initialization
try:
    APP_ID = st.secrets["APP_ID"]
    wolfram_client = wolframalpha.Client(APP_ID)
except KeyError:
    st.error("Error: APP_ID not found in Streamlit secrets. Please configure it.")
    wolfram_client = None
except Exception as e:
    st.error(f"Error initializing Wolfram Alpha client: {e}")
    wolfram_client = None

# APP logo
logo = "https://avatars.githubusercontent.com/u/175069629?v=4"

## Functions for the assistant

def search_wikipedia(query):
    try:
        result = wikipedia.summary(query, sentences=2)
        return result
    except wikipedia.exceptions.DisambiguationError as e:
        return "Multiple meanings detected. Please specify: " + ", ".join(e.options[:5])
    except wikipedia.exceptions.PageError:
        return "No results found on Wikipedia."
    except Exception as e:
        logging.error(f"Wikipedia error: {e}")
        return "An error occurred while searching Wikipedia."

def query_wolfram_alpha(query):
    if wolfram_client is None:
        return "Wolfram Alpha is not configured."
    try:
        res = wolfram_client.query(query)
        return next(res.results).text
    except Exception as e:
        logging.error(f"Wolfram Alpha error: {e}")
        return "An error occurred while querying Wolfram Alpha."

def query_google_gemini(query, context=""):
    if model is None:
        return "Gemini is not configured."
    try:
        conversation_input = context + f"\nUser: {query}\nAssistant:"
        response = model.generate_content(conversation_input)
        return response.text
    except genai.types.generation.GenerationError as e:
        logging.error(f"Gemini Generation Error: {e}")
        return f"Gemini encountered an error during generation: {e.message}"
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return f"An error occurred while fetching from Google Gemini: {e}"

def extract_text_from_file(file):
    try:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            text = "".join(page.extract_text() for page in pdf_reader.pages)
            return text
        elif file.type == "text/plain":
            text = file.read().decode("utf-8")
            return text
        else:
            return None
    except Exception as e:
        logging.error(f"File processing error: {e}")
        return None

# Streamlit App
st.set_page_config(page_title="Projekt S.A.N.A", page_icon=logo, layout="wide")

# Sidebar
with st.sidebar:
    st.title("S.A.N.A Settings")
    st.markdown("⚙️ **Customize your assistant experience (coming soon!)**")
    st.markdown("---")
    st.markdown("Use the features below to interact with S.A.N.A:")
    st.markdown("1. Wikipedia Search\n2. Wolfram Alpha Queries\n3. Google Gemini Chat\n4. PDF/TXT Summary")

# Main App

# Logo and Title
st.markdown(f"<h1><img src='{logo}' width=70 style='display:inline-block; margin-right:15px'></img><b>Projekt S.A.N.A:</b></h1>", unsafe_allow_html=True)

# Description
st.markdown("""
**S.A.N.A** is a secure, autonomous, and non-intrusive virtual assistant. 
Feel free to ask me anything! 😊
""")
st.markdown("---")

# Initialize session variables
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "context" not in st.session_state:
    st.session_state["context"] = ""

# Feature Selection
feature = st.selectbox("Select a feature to use:",
                       ["General Chat", "Wikipedia Search", "Wolfram Alpha Queries", "PDF/TXT Summary"], index=0)

# Display Chat History
st.markdown("### 💬 Chat History")
st.write("---")
for sender, message in st.session_state["chat_history"]:
    if sender == "You":
        st.markdown(f"**🧑‍💻 You:** {message}")
    elif sender == "S.A.N.A":
        st.markdown(f"<img src='{logo}' width=20 style='display:inline-block; margin-right:10px'></img><b>S.A.N.A:</b> {message}", unsafe_allow_html=True)
    else:
        st.markdown(f"**❗Unknown Sender:** {message}")

# User Input or File Upload
st.write("---")

if feature == "PDF/TXT Summary":
    uploaded_file = st.file_uploader("📂 Upload a PDF or TXT file:", type=["pdf", "txt"])
    if uploaded_file is not None:
        file_text = extract_text_from_file(uploaded_file)
        if file_text:
            st.write("### Extracted Text:")
            st.text_area("Preview of the file content:", value=file_text, height=200)
            if st.button("Summarize File"):
                summary = query_google_gemini(file_text, st.session_state["context"])
                st.markdown("### Summary:")
                st.write(summary)
                st.session_state["chat_history"].append(("You", "Uploaded a file for summary"))
                st.session_state["chat_history"].append(("S.A.N.A", summary))
                st.experimental_rerun()
        else:
            st.error("Unsupported file type or failed to extract text.")
else:
    user_input = st.text_input("💬 Type your query below:", placeholder="Ask anything...", key="user_input")
    if st.button("Send"):
        if user_input:
            st.session_state["chat_history"].append(("You", user_input))
            try:
                if feature == "Wikipedia Search":
                    response = search_wikipedia(user_input)
                elif feature == "Wolfram Alpha Queries":
                    response = query_wolfram_alpha(user_input)
                elif feature == "General Chat":
                    response = query_google_gemini(user_input, st.session_state["context"])
                else:
                    response = "Invalid feature selected."
                st.session_state["chat_history"].append(("S.A.N.A", response))
                st.session_state["context"] += f"User: {user_input}\nAssistant: {response}\n"
            except Exception as e:
                logging.error(f"Main processing error: {e}")
                st.error(f"An unexpected error occurred: {e}")
                st.session_state["chat_history"].append(("S.A.N.A", "An unexpected error occurred. Please check the logs."))
            st.experimental_rerun()

# Clear History Button
st.write("---")
if st.button("Clear Chat History"):
    st.session_state["chat_history"] = []
    st.session_state["context"] = ""
    st.success("Chat history cleared!")
