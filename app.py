import streamlit as st
import wikipedia
import wolframalpha
import google.generativeai as genai
import requests
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Google Gemini API Configuration
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

# WolframAlpha Configuration
try:
    APP_ID = st.secrets["APP_ID"]
    wolfram_client = wolframalpha.Client(APP_ID)
except KeyError:
    st.error("Error: APP_ID not found in Streamlit secrets. Please configure it.")
    wolfram_client = None
except Exception as e:
    st.error(f"Error initializing Wolfram Alpha client: {e}")
    wolfram_client = None

# Hugging Face API Configuration for Image Description
HF_API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
HF_API_KEY = st.secrets["HF_API_KEY"]
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# Functions
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
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        payload = buffer.read()
        response = requests.post(HF_API_URL, headers=headers, data=payload)
        response.raise_for_status()
        result = response.json()
        return result[0]["generated_text"]
    except Exception as e:
        logging.error(f"Image description error: {e}")
        return "Failed to describe the image."

# App Configuration
logo = "https://avatars.githubusercontent.com/u/175069629?v=4"
st.set_page_config(page_title="Projekt S.A.N.A", page_icon=logo, layout="wide")

# Sidebar
with st.sidebar:
    st.image(logo, width=120)
    st.title("S.A.N.A Settings")
    st.markdown("⚙️ **Customize your assistant experience (coming soon!)**")
    st.markdown("---")
    st.markdown("Available features:")
    st.markdown("1. Wikipedia Search\n2. Wolfram Alpha Queries\n3. General Chat\n4. Image Description")

# Initialize session variables
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "context" not in st.session_state:
    st.session_state["context"] = ""

# Main Layout
st.image(logo, width=125)
st.title("Projekt S.A.N.A")
st.markdown("""
Welcome to **S.A.N.A**: A secure, autonomous, and non-intrusive assistant. Select a feature below to interact.
""")

# Feature Selection
feature = st.selectbox("Select a feature to use:",
                       ["General Chat", "Wikipedia Search", "Wolfram Alpha Queries", "Image Description"], index=0)

# Display Chat History
if feature != "Image Description":
    st.markdown("### 💬 Chat History")
    st.write("---")
    for sender, message in st.session_state["chat_history"]:
        if sender == "You":
            st.markdown(f"**🧑‍💻 You:** {message}")
        elif sender == "S.A.N.A":
            st.markdown(f"🤖 **S.A.N.A:** {message}")
        else:
            st.markdown(f"**❗Unknown Sender:** {message}")

# User Input
if feature != "Image Description":
    user_input = st.text_input("💬 Type your query below:", placeholder="Ask anything...")
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
    if st.button("Clear Chat History"):
        st.session_state["chat_history"] = []
        st.session_state["context"] = ""
        st.success("Chat history cleared!")

# Image Description Feature
if feature == "Image Description":
    st.markdown("### 🖼️ Image Description")
    st.markdown("Upload an image or take a picture to generate a description of it.")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Generating description..."):
            description = describe_image(image)
            st.success("Description Generated!")
            st.write(description)

    st.markdown("---")
    st.write("Or, take a picture using your camera:")
    picture = st.camera_input("Take a Picture")
    if picture:
        image = Image.open(picture)
        st.image(image, caption="Captured Image", use_column_width=True)
        with st.spinner("Generating description..."):
            description = describe_image(image)
            st.success("Description Generated!")
            st.write(description)

# Footer
st.write("---")
st.markdown("Powered by Hugging Face, Wolfram Alpha, Wikipedia, and Google Gemini.")
