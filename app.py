import streamlit as st
import wikipedia
import wolframalpha
import google.generativeai as genai
import base64
import requests
from PIL import Image
import io

# Ensure set_page_config is the first command
st.set_page_config(page_title="Projekt S.A.N.A", page_icon="🤖", layout="wide")

# Set API Keys
GENAI_API_KEY = st.secrets["GENAI_API_KEY"]
APP_ID = st.secrets["APP_ID"]
HF_API_KEY = st.secrets["HF_API_KEY"]

# Configure API Clients
genai.configure(api_key=GENAI_API_KEY)
system_prompt = "You are S.A.N.A (Secure Autonomous Non-Intrusive Assistant), a smart, privacy-respecting AI."
model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=[system_prompt])
wolfram_client = wolframalpha.Client(APP_ID)

# Hugging Face API Endpoints
HF_IMAGE_MODEL = "Salesforce/blip-image-captioning-large"
HF_GEN_MODEL = "stabilityai/stable-diffusion-2"

# Sidebar for Feature Selection and Settings
with st.sidebar:
    st.title("S.A.N.A Settings")
    feature = st.selectbox(
        "Select a feature:",
        ["General Chat", "Wikipedia Search", "Wolfram Alpha Queries", "Image Description", "Image Generation"]
    )
    
    # Clear Chat History Button
    if st.button("🗑️ Clear Chat History"):
        st.session_state["chat_history"] = []
        st.experimental_rerun()

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display Chat History
st.markdown("## 💬 Chat History")
for sender, message in st.session_state["chat_history"]:
    st.markdown(f"**{sender}:** {message}")
st.write("---")

# User Input
user_input = st.text_input("💬 Type your query:", placeholder="Ask anything...", key="user_input")

if st.button("Send") and user_input:
    st.session_state["chat_history"].append(("You", user_input))
    
    # Wikipedia Search
    if feature == "Wikipedia Search":
        try:
            response = wikipedia.summary(user_input, sentences=2)
        except wikipedia.exceptions.DisambiguationError as e:
            response = f"Multiple results found: {', '.join(e.options[:5])}"
        except wikipedia.exceptions.PageError:
            response = "No Wikipedia page found."
    
    # Wolfram Alpha Queries
    elif feature == "Wolfram Alpha Queries":
        try:
            res = wolfram_client.query(user_input)
            response = next(res.results).text
        except:
            response = "Wolfram Alpha couldn't process your query."
    
    # General Chat (Gemini)
    elif feature == "General Chat":
        response = model.generate_content(user_input).text
    
    # Image Description
    elif feature == "Image Description":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_image:
            image_bytes = uploaded_image.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}",
                headers=headers,
                json={"image": image_base64}
            ).json()[0]["generated_text"]

    # Image Generation
    elif feature == "Image Generation":
        prompt = user_input
        if prompt:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{HF_GEN_MODEL}",
                headers=headers,
                json={"inputs": prompt}
            )
            image_bytes = response.content
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, caption="Generated Image")
            response = "Image generated successfully!"
    
    st.session_state["chat_history"].append(("S.A.N.A", response))
    st.experimental_rerun()
