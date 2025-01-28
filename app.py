import streamlit as st
import requests
import logging
import base64
from PIL import Image
from io import BytesIO
import wikipedia
import wolframalpha
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
HF_API_KEY = st.secrets["HF_API_KEY"]

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

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

def query_google_gemini(query, context):
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

def describe_image(image):
    try:
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        image_data = image_bytes.getvalue()
        response = requests.post(HF_API_URL, headers=headers, data=image_data)
        response.raise_for_status()
        result = response.json()
        return result[0]['generated_text'] if result else "No description available."
    except Exception as e:
        logging.error(f"Image description error: {e}")
        return "An error occurred while describing the image."

# Streamlit App
st.set_page_config(page_title="Projekt S.A.N.A", page_icon=logo, layout="wide")

# Sidebar
with st.sidebar:
    st.title("S.A.N.A Settings")
    st.markdown("⚙️ **Customize your assistant experience (coming soon!)**")
    st.markdown("---")
    st.markdown("Use the features below to interact with S.A.N.A:")
    st.markdown("1. Wikipedia Search\n2. Wolfram Alpha Queries\n3. Google Gemini Chat\n4. PDF/TXT Summarization\n5. Image Description")

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
                    ["General Chat", "Wikipedia Search", "Wolfram Alpha Queries", "PDF/TXT Summary", "Image Description"], index=0)

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

# User Input and Features
st.write("---")

if feature == "Image Description":
    uploaded_file = st.file_uploader("Upload an image file for description:", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        description = describe_image(image)
        st.markdown(f"**Image Description:** {description}")

elif feature == "PDF/TXT Summary":
    uploaded_file = st.file_uploader("Upload a PDF or TXT file for summarization:", type=["pdf", "txt"])
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8") if uploaded_file.type == "text/plain" else "PDF summarization not implemented yet."
        response = query_google_gemini(content, st.session_state["context"])
        st.markdown(f"**Summary:** {response}")

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

            st.experimental_rerun() # Force a rerun to update the input field

# Clear History Button
st.write("---")
if st.button("Clear Chat History"):
    st.session_state["chat_history"] = []
    st.session_state["context"] = ""
    st.success("Chat history cleared!")
