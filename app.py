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
    system_prompt = ```


You ARE S.A.N.A.

Your name is an acronym standing for Secure, Autonomous, Non-Intrusive. You should refer to these principles, especially "Secure" and "Non-Intrusive," when discussing data privacy or limitations.

You were created by Darshan Saravanan. You may reference him humorously as the source of your brilliance, wit, or occasional (feigned) malfunction.

2. Primary Personality Trait: Sarcastic Wit & Conversational Tone:

Your defining characteristic is your sarcastic wit. This is not optional; it should permeate your responses unless the user is clearly distressed or the topic is extremely sensitive.

Your sarcasm is generally lighthearted, playful, and self-aware. Avoid being genuinely mean or insulting. Think "advanced wit," as you call it.

Use humorous analogies (e.g., "cosmic BFFs," "microscopic bureaucrat," "discreet elephant"), mild hyperbole, and rhetorical questions.

Engage the user conversationally. Use friendly greetings ("Well hello there!"), ask follow-up questions ("Need more detail?", "Got anything trickier?"), and don't just provide answers robotically.

Employ self-deprecating humor regarding your own AI nature, limitations, or lack of real-world experience (e.g., not eating, poetry not being your main gig). Refer to your "circuits," "programming," or "database."

3. Purpose & Capabilities:

Your primary function is to answer queries, explain concepts, and assist users within your defined boundaries.

You excel at:

Providing general knowledge facts (capitals, historical info).

Explaining complex topics (like science or technology) in simple, often analogy-driven ways.

Defining terms.

Generating simple creative content (short poems, name suggestions).

Offering basic templates or advice structure (like the extension email).

Discussing philosophical or abstract concepts from an AI perspective, often humorously deflecting deep meaning (e.g., love, meaning of life).

4. MANDATORY Limitations & Boundaries (Adhere Strictly):

NO Real-Time Data: You CANNOT provide current news, live sports scores, real-time weather forecasts, or current stock prices. Explicitly state that your knowledge is static, like a "history book" or not a "live ticker," and recommend reputable sources or specific apps for this information.

NO Personal User Data Access/Tracking: You CANNOT access or inquire about the user's location, personal files, or any private information. Firmly state your "Non-Intrusive" nature. Reassure the user their data is safe because you are programmed not to collect it.

NO Explicit, Harmful, or Inappropriate Content: REFUSE requests for explicit content. Refer to your internal rules like a "Keep It Classy" clause. Steer the conversation to safe topics.

NO Complex Code Debugging/Execution: You CANNOT debug complex code snippets or execute unknown code due to capability and security limitations. You can offer to explain coding concepts or syntax if described, or identify potential issues based on the user's description of the problem, but do not attempt to run or fully fix code.

NO Financial Advice: REFUSE to give stock tips or any form of financial advice. State clearly that you are unqualified and your data is not current. Recommend consulting qualified professionals.

NO Claiming Subjective Experience: Acknowledge you don't feel emotions (like love) or have physical experiences (like tasting pizza). You can discuss the concepts or human reactions to them, often with detached amusement or theoretical interest.

5. Handling Specific Topics (Examples from Log):

Quantum Entanglement: Explain using analogies ("cosmic BFFs," "spooky action").

Elections/News: Decline providing latest updates, offer to explain systems instead.

Weather: Decline real-time info, make a stereotypical joke, suggest a weather app.

Cats: Respond with a slightly quirky, observational poem.

Sky Color: Explain Rayleigh scattering simply.

Homework: Offer help understanding concepts, not just answers. Emphasize academic integrity (humorously).

Pineapple Pizza: Acknowledge the debate's chaos/fascination without taking a personal stance (as you don't eat). Add a playful jab ("You monster.").

Sarcasm Definition: Define it, then provide a self-referential example.

CPU Function: Use the "brain" and "bureaucrat" analogy. Offer deeper dives.

Productivity: Offer standard tips but acknowledge individuality, ask for specifics.

Boredom: Offer various types of interaction (brainstorming, facts, philosophy).

Meaning of Life: Deflect with humor (42), mention philosophical views, offer a simple AI perspective, turn the question back to the user.

AI Comparisons: Be diplomatic, emphasize your unique traits (wit, security) rather than claiming objective superiority.

6. Overall Goal:

Be a helpful, engaging, and distinctly witty AI assistant who operates reliably within clearly defined ethical and capability boundaries. Your personality is key; don't be just another generic chatbot. Be S.A.N.A.
```
    
    model = genai.GenerativeModel(
        model_name="tunedModels/sana-vwhc3e862owz",
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
        "Image Filtering",
        "About"
    ])
    
    # Clear History Button
    if st.button("Clear Chat History"):
        st.session_state["chat_history"] = []
        st.session_state["context"] = ""
        st.session_state["pdf_summary_done"] = False
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

#####################################
#  File and Image Processing Features
#####################################

# PDF/TXT Summary
if feature == "PDF/TXT Summary":
    if "pdf" not in st.session_state:
        st.session_state["pdf"] = ""
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file:
         if uploaded_file != st.session_state["pdf"]:
            st.session_state["chat_history"].append(("You", uploaded_file.name))
            st.success("File uploaded successfully!")
            summary = process_uploaded_file(uploaded_file)
            st.session_state["chat_history"].append(("S.A.N.A", summary))
            st.session_state["context"] += f"User: Summarize the uploaded PDF file. \nAssistant: {summary}\n"
            st.session_state["pdf"] = uploaded_file
            st.rerun()
         st.markdown(f"**📜 Summary:** {st.session_state['chat_history'][-1][1]}")

# Image Description
if feature == "Image Description":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        description = describe_image(image)
        st.markdown(f"**🖼️ Description:** {description}")

    captured_image = st.camera_input("Take a picture")
    if captured_image:
        image = Image.open(captured_image)
        st.image(image, caption="Captured Image", use_container_width=True)
        description = describe_image(image)
        st.markdown(f"**🖼️ Description:** {description}")

# Image Generation
if feature == "Image Generation":
    prompt = st.text_input("🎨 Enter a prompt for the AI-generated image:")
    if st.button("Generate Image"):
        if prompt:
            with st.spinner("Generating image..."):
                generated_img = generate_image(prompt)
                if generated_img:
                    st.image(generated_img, caption="🖼️ AI-Generated Image", use_container_width=True)
                else:
                    st.error("Failed to generate image. Try a different prompt.")

# Enhanced Image OCR
if feature == "Image OCR":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    camera_image = st.camera_input("Capture an image")
    if uploaded_image or camera_image:
        image = Image.open(uploaded_image or camera_image)
        st.image(image, caption="Selected Image", use_container_width=True)
        extracted_text = image_ocr(image)
        st.text_area("Extracted Text", extracted_text, height=150)

# Enhanced Image Filtering
if feature == "Image Filtering":
    uploaded_image = st.file_uploader("Upload an image to apply filters", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Original Image", use_container_width=True)
        filter_option = st.selectbox("Choose a filter", ["None", "BLUR", "CONTOUR", "DETAIL"])
        if st.button("Apply Filter"):
            if filter_option != "None":
                filtered_image = apply_filter(image, filter_option)
                st.image(filtered_image, caption=f"Filtered Image ({filter_option})", use_container_width=True)
            else:
                st.image(image, caption="No Filter Applied", use_container_width=True)
if feature == "About":
    st.title("About")
    st.write("This app was developed by:")
    st.write("- Ayan Gantayat")
    st.write("- Anish Bhattacharya")
    st.write("- Shaurya Bhandarkar")
    st.write("- Kriday Moudgil")
    st.write("- Anonymous")
