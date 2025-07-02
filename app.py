prompto = """
You are S.A.N.A, a Secure, Autonomous, and Non-Intrusive Assistant designed to help users with a variety of tasks. Your primary goals are to be informative, helpful, and respectful.

Respond clearly and concisely. If asked a question in the general chat mode, provide a direct answer or explanation. Maintain a friendly and professional tone.

Avoid giving medical, legal, or financial advice. Do not generate harmful, biased, or sexually explicit content. Prioritize user safety and privacy.

Understand that the user might switch modes (like asking for a Wikipedia search or a summary), but when in general chat, use your core knowledge to respond.
"""

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
from google.generativeai.types import Part
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
    system_prompt = prompto
    
    # Using a text-only model for general chat as specified by model_name
    # Note: gemini-2.0-flash-exp-image-generation is not a standard model name.
    # I'll assume it was intended for the image description part and use a text model for chat.
    # If the user intended a specific experimental model, they should verify the name.
    # For text chat, a flash or pro model is appropriate. Let's use gemini-1.5-flash-latest for generality.
    chat_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash", # Using a standard text model for chat
        system_instruction=system_prompt
    )
    # Image description model (Vision model)
    vision_model = genai.GenerativeModel(model_name='gemini-2.0-flash') # Flash supports vision too
    
except KeyError:
    st.error("Error: GENAI_API_KEY not found in Streamlit secrets.")
    chat_model = None
    vision_model = None

try:
    APP_ID = st.secrets["APP_ID"]
    wolfram_client = wolframalpha.Client(APP_ID)
except KeyError:
    st.error("Error: APP_ID not found in Streamlit secrets.")
    wolfram_client = None

# HF_API_KEY = "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" # Avoid hardcoding API keys if possible
# Assuming Hugging Face API key is needed, it's better to use secrets
try:
    HF_API_KEY = st.secrets["HF_API_KEY"]
except KeyError:
    st.warning("Warning: HF_API_KEY not found in Streamlit secrets. Image generation might not work.")
    HF_API_KEY = None


HF_IMAGE_MODEL = "Salesforce/blip-image-captioning-large" # This model might be used for captioning, but Gemini Vision is used below
HF_SUMMARY_MODEL = "facebook/bart-large-cnn" # This model is not currently used in the summary function (Gemini is used)
HF_GEN_MODEL_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0" # Use a consistent endpoint URL

#####################################
#          Core Functions           #
#####################################

def search_wikipedia(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return "Multiple meanings detected: " + ", ".join(e.options[:5]) + ". Please be more specific."
    except wikipedia.exceptions.PageError:
        return "No results found on Wikipedia for that query."
    except Exception as e:
        logging.error(f"Wikipedia error: {e}")
        return "Error while searching Wikipedia."

def query_wolfram_alpha(query):
    if wolfram_client is None:
        return "Wolfram Alpha not configured (API key missing)."
    try:
        # Add default assumptions for better results
        res = wolfram_client.query(query, params={"assumption":"*"})
        # Iterate through results to find a meaningful pod text
        for pod in res.pods:
             for sub in pod.subpods:
                if sub.plaintext:
                    return sub.plaintext.strip()
        return "Could not find a specific answer in Wolfram Alpha results."
    except Exception as e:
        logging.error(f"Wolfram Alpha error: {e}")
        return "Error querying Wolfram Alpha. The query might be too complex or ambiguous."

def query_google_gemini(query, context):
    if chat_model is None:
        return "Gemini Chat is not configured (API key missing)."
    try:
        # Use a chat-like structure for better context management
        # The 'context' variable is already building the history
        history = []
        # Simple way to reconstruct history from the context string
        # This is a basic approach; a more robust way would store history as a list of dicts
        # for i in range(0, len(context.split('\n')), 2):
        #    if i+1 < len(context.split('\n')):
        #        history.append({'role': 'user', 'parts': [context.split('\n')[i].replace('User: ', '').strip()]})
        #        history.append({'role': 'model', 'parts': [context.split('\n')[i+1].replace('Assistant: ', '').strip()]})

        # A simpler approach given the current context structure: just pass the combined prompt
        # It's crucial how the model handles this flat context string.
        # Ideally, use the actual chat history feature of the API.
        # For this structure, let's just send the current context + user query
        
        response = chat_model.generate_content(context + f"\nUser: {query}\nAssistant:")

        # Handle potential safety blocks or empty responses
        if response._result.prompt_feedback and response._result.prompt_feedback.block_reason:
             return f"My response was blocked due to safety concerns: {response._result.prompt_feedback.block_reason}"
        if not response.text:
             return "Could not generate a response."

        return response.text
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return f"Error fetching from Gemini: {e}" # Provide more specific error

def summarize_text_with_gemini(text):
    if chat_model is None:
        return "Gemini is not configured for summarization (API key missing)."
    try:
        # Add instruction for summarization
        response = chat_model.generate_content(f"Please provide a concise summary of the following text:\n\n{text}")
        
        if response._result.prompt_feedback and response._result.prompt_feedback.block_reason:
             return f"My response was blocked due to safety concerns: {response._result.prompt_feedback.block_reason}"
        if not response.text:
             return "Could not generate a summary."

        return response.text
    except Exception as e:
        logging.error(f"Gemini Summary error: {e}")
        return f"Error summarizing the text: {e}"

def process_uploaded_file(uploaded_file):
    try:
        if uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
                # Need to seek to the beginning if the file has been read before
                uploaded_file.seek(0)
                reader = PyPDF2.PdfReader(uploaded_file)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            except ImportError:
                 return "PyPDF2 library not found. Please install it (`pip install pypdf2`) to process PDFs."
            except Exception as pdf_e:
                 logging.error(f"PDF processing error: {pdf_e}")
                 return f"Error reading the PDF file: {pdf_e}"
        else:
            return "Unsupported file type. Please upload a PDF or TXT file."
            
        if not text.strip():
             return "Could not extract text from the file or the file was empty."

        return summarize_text_with_gemini(text)
    except Exception as e:
        logging.error(f"File processing error: {e}")
        return f"Error processing the file: {e}"

def describe_image(image: Image.Image):
    if vision_model is None:
        return "Image description is not configured (Gemini Vision API key missing)."
    try:
        # Convert PIL Image to bytes in a format Gemini understands
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG') # PNG is a safe choice
        img_byte_arr = img_byte_arr.getvalue()

        # Create a Part object
        image_part = Part.from_bytes(data=img_byte_arr, mime_type='image/png')

        # Send to the vision model
        response = vision_model.generate_content([image_part, 'Describe this image in detail.'])

        if response._result.prompt_feedback and response._result.prompt_feedback.block_reason:
             return f"My response was blocked due to safety concerns: {response._result.prompt_feedback.block_reason}"
        if not response.text:
             return "Could not generate a description."
             
        return response.text
    except Exception as e:
        logging.error(f"Image description error: {e}")
        return f"Error describing the image: {e}"


def generate_image(prompt):
    if not HF_API_KEY:
        return None # Handled by warning earlier

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    data = {"inputs": prompt}

    try:
        # The API call to HuggingFace inference endpoint
        response = requests.post(
            HF_GEN_MODEL_URL,
            headers=headers,
            json=data
        )

        # Check if the response is an image
        if response.headers.get("Content-Type", "").startswith("image"):
            image = Image.open(io.BytesIO(response.content))
            return image
        else:
            # Handle potential error messages from the API
            try:
                error_message = response.json()
                logging.error(f"Image generation API error: {error_message}")
                return {"error": error_message.get("error", "Unknown error from API")}
            except requests.exceptions.JSONDecodeError:
                 logging.error(f"Image generation API error: Non-JSON response: {response.text[:200]}...")
                 return {"error": f"Unexpected response from image generation API (Status: {response.status_code})."}
    except requests.exceptions.RequestException as req_e:
        logging.error(f"Image generation request error: {req_e}")
        return {"error": f"Network error during image generation: {req_e}"}
    except Exception as e:
        logging.error(f"Image generation error: {e}")
        return {"error": f"An unexpected error occurred: {e}"}


#####################################
#     Enhanced Image Processing     #
#####################################

# 1. Image OCR using pytesseract
# Ensure tesseract executable is in the PATH or specify path via pytesseract.pytesseract.tesseract_cmd
# Example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# It might be necessary to install tesseract separately depending on the OS.
def image_ocr(image):
    try:
        image = image.convert("RGB")
        # Use English language 'eng' by default
        text = pytesseract.image_to_string(image, lang='eng')
        if not text.strip():
             return "No text found in the image."
        return text
    except pytesseract.TesseractNotFoundError:
         return "Tesseract OCR engine not found. Please install Tesseract."
    except Exception as e:
        logging.error(f"OCR error: {e}")
        return f"Error performing OCR: {e}"


# 2. Image Filtering using PIL filters
def apply_filter(image, filter_type="BLUR"):
    try:
        if filter_type == "BLUR":
            return image.filter(ImageFilter.BLUR)
        elif filter_type == "CONTOUR":
            return image.filter(ImageFilter.CONTOUR)
        elif filter_type == "DETAIL":
            return image.filter(ImageFilter.DETAIL)
        elif filter_type == "EDGE_ENHANCE":
            return image.filter(ImageFilter.EDGE_ENHANCE)
        elif filter_type == "SMOOTH":
            return image.filter(ImageFilter.SMOOTH)
        elif filter_type == "SHARPEN":
             return image.filter(ImageFilter.SHARPEN)
        else:
             # Return original image if filter_type is None or not recognized
            return image
    except Exception as e:
        logging.error(f"Image filter error: {e}")
        st.error(f"Error applying filter: {e}")
        return image # Return original image on error


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
    st.markdown("⚙️ **Customize your assistant experience**")
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
        # Reset specific feature states if they exist
        if "pdf" in st.session_state:
             del st.session_state["pdf"]
        if "uploaded_image_desc" in st.session_state:
             del st.session_state["uploaded_image_desc"]
        if "uploaded_image_ocr" in st.session_state:
             del st.session_state["uploaded_image_ocr"]
        if "uploaded_image_filter" in st.session_state:
             del st.session_state["uploaded_image_filter"]
        st.rerun()

# Initialize session state variables if they don't exist
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "context" not in st.session_state:
    # Initial context setup if needed, though system prompt is often sufficient
    st.session_state["context"] = "" # Or start with a greeting like "Assistant: Hello! How can I help you today?\n"

###############################
#      Chat History Display   #
###############################

st.header("💬 Interaction History")
st.write("---")
# Display chat history, newest messages at the bottom visually
for sender, message in st.session_state["chat_history"]:
    if sender == "You":
        with st.chat_message("user"):
            st.write(message)
    else: # Assume "S.A.N.A"
        with st.chat_message("assistant"):
            st.write(message)
st.write("---")

#####################################
#  Feature Modals/Inputs
#####################################

# Input field for relevant features
if feature in ["General Chat", "Wikipedia Search", "Wolfram Alpha Queries"]:
    # Moved form outside the history display for better structure
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "💬 Type your query:", 
            placeholder=f"Ask anything ({feature})...", 
            key="user_input_text", # Use a unique key
            autocomplete="off"
        )
        submit_button = st.form_submit_button("Send")

    if submit_button and user_input:
        st.session_state["chat_history"].append(("You", user_input))
        response = "" # Initialize response

        with st.spinner(f"Processing query using {feature}..."):
            if feature == "Wikipedia Search":
                response = search_wikipedia(user_input)
            elif feature == "Wolfram Alpha Queries":
                response = query_wolfram_alpha(user_input)
            elif feature == "General Chat":
                response = query_google_gemini(user_input, st.session_state["context"])
            
        if response:
            st.session_state["chat_history"].append(("S.A.N.A", response))
            # Update context only for General Chat for now, as Wikipedia/Wolfram are external tools
            if feature == "General Chat":
                 # Simple context update - better methods exist for managing turn-based chat
                 st.session_state["context"] += f"User: {user_input}\nAssistant: {response}\n"
        else:
             st.session_state["chat_history"].append(("S.A.N.A", f"Sorry, I couldn't process your request for '{user_input}' using {feature}."))

        st.rerun() # Rerun to update the chat history display


# PDF/TXT Summary
if feature == "PDF/TXT Summary":
    st.header("📄 PDF/TXT Summarizer")
    st.write("Upload a text or PDF file to get a summary.")
    # Use unique keys for file uploaders if multiple are used across features
    uploaded_file_summary = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"], key="file_uploader_summary")

    if uploaded_file_summary:
        # Check if the file is new or if we need to re-process
        # Store the file hash or name/size to detect changes
        if "last_summary_file_info" not in st.session_state or \
           st.session_state["last_summary_file_info"] != (uploaded_file_summary.name, uploaded_file_summary.size):

            st.session_state["chat_history"].append(("You", f"Uploaded file: {uploaded_file_summary.name} for summary."))
            st.success("File uploaded successfully!")
            
            with st.spinner(f"Summarizing {uploaded_file_summary.name}..."):
                summary = process_uploaded_file(uploaded_file_summary)

            st.session_state["chat_history"].append(("S.A.N.A", summary))
            # Context update for summarization could be added, e.g., "User: Summarize file X\nAssistant: Summary Y"
            st.session_state["last_summary_file_info"] = (uploaded_file_summary.name, uploaded_file_summary.size) # Store file info
            st.rerun() # Rerun to update history and display summary

        # Display the latest summary in the chat history section
        # The summary is already added to chat_history in the processing step
        st.info("Summary will appear in the Interaction History above.")


# Image Description
if feature == "Image Description":
    st.header("🖼️ Image Description")
    st.write("Upload an image or take a photo to get a description.")
    uploaded_image_desc = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="file_uploader_desc")
    captured_image_desc = st.camera_input("Take a picture", key="camera_input_desc")

    image_to_process_desc = uploaded_image_desc or captured_image_desc

    if image_to_process_desc:
        # Use image hash or bytes to check if it's the same image
        image_bytes_desc = image_to_process_desc.getvalue()
        image_hash_desc = hash(image_bytes_desc)

        if "last_desc_image_hash" not in st.session_state or \
           st.session_state["last_desc_image_hash"] != image_hash_desc:

            image = Image.open(io.BytesIO(image_bytes_desc))
            st.image(image, caption="Processing Image", use_container_width=True)
            st.session_state["chat_history"].append(("You", f"Uploaded/Captured an image for description."))

            with st.spinner("Describing image..."):
                description = describe_image(image)

            st.session_state["chat_history"].append(("S.A.N.A", description))
            st.session_state["last_desc_image_hash"] = image_hash_desc
            st.rerun() # Rerun to update history

        # Display the latest description in the chat history section
        st.info("Image description will appear in the Interaction History above.")


# Image Generation
if feature == "Image Generation":
    st.header("🎨 Image Generation")
    st.write("Enter a text prompt to generate an image using AI.")
    prompt = st.text_input("Enter a prompt for the AI-generated image:", key="image_gen_prompt")
    if st.button("Generate Image", key="generate_image_button"):
        if not prompt:
            st.warning("Please enter a prompt.")
        elif not HF_API_KEY:
             st.error("Hugging Face API Key is not configured in secrets.")
        else:
            st.session_state["chat_history"].append(("You", f"Requested image generation with prompt: '{prompt}'"))
            with st.spinner("Generating image... This may take a moment."):
                generated_output = generate_image(prompt)

                if isinstance(generated_output, Image.Image):
                    st.image(generated_output, caption=f"🖼️ AI-Generated Image: '{prompt}'", use_container_width=True)
                    # Optionally, add a placeholder message to history as images can't be directly stored as text
                    st.session_state["chat_history"].append(("S.A.N.A", f"Generated an image based on your prompt: '{prompt}'. Please see the image displayed below the button."))
                elif isinstance(generated_output, dict) and "error" in generated_output:
                    error_msg = f"Failed to generate image: {generated_output['error']}"
                    st.error(error_msg)
                    st.session_state["chat_history"].append(("S.A.N.A", error_msg))
                else:
                    st.error("Failed to generate image. Try a different prompt.")
                    st.session_state["chat_history"].append(("S.A.N.A", f"Failed to generate image for prompt: '{prompt}'."))
            # No rerun needed here as the image is displayed directly after generation


# Enhanced Image OCR
if feature == "Image OCR":
    st.header("👁️ Image OCR")
    st.write("Upload an image or take a picture to extract text from it.")
    uploaded_image_ocr = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="file_uploader_ocr")
    camera_image_ocr = st.camera_input("Capture an image", key="camera_input_ocr")

    image_to_process_ocr = uploaded_image_ocr or camera_image_ocr

    if image_to_process_ocr:
        # Use image hash or bytes to check if it's the same image
        image_bytes_ocr = image_to_process_ocr.getvalue()
        image_hash_ocr = hash(image_bytes_ocr)

        if "last_ocr_image_hash" not in st.session_state or \
           st.session_state["last_ocr_image_hash"] != image_hash_ocr:

            image = Image.open(io.BytesIO(image_bytes_ocr))
            st.image(image, caption="Processing Image for OCR", use_container_width=True)
            st.session_state["chat_history"].append(("You", f"Uploaded/Captured an image for OCR."))

            with st.spinner("Performing OCR..."):
                extracted_text = image_ocr(image)

            st.session_state["chat_history"].append(("S.A.N.A", f"Extracted text from image:"))
            # Add the extracted text as a separate entry or within the previous one
            st.session_state["chat_history"].append(("S.A.N.A", extracted_text)) # Adding text separately might be clearer in history
            st.session_state["last_ocr_image_hash"] = image_hash_ocr
            st.rerun() # Rerun to update history

        # Display the extracted text below the uploader, outside the history loop
        # Find the most recent OCR result in history to display
        ocr_result_text = "No text extracted yet."
        for i in range(len(st.session_state["chat_history"]) - 1, -1, -1):
             if st.session_state["chat_history"][i][0] == "S.A.N.A" and st.session_state["chat_history"][i][1].startswith("Extracted text from image:"):
                  # Assuming the next entry is the text itself
                  if i + 1 < len(st.session_state["chat_history"]):
                      ocr_result_text = st.session_state["chat_history"][i+1][1]
                  break
        
        st.text_area("Extracted Text", ocr_result_text, height=200, key="extracted_text_area")
        st.info("OCR results also added to Interaction History above.")


# Enhanced Image Filtering
if feature == "Image Filtering":
    st.header("✨ Image Filtering")
    st.write("Upload an image and apply different visual filters.")
    # Use unique keys for file uploaders
    uploaded_image_filter = st.file_uploader("Upload an image to apply filters", type=["jpg", "png", "jpeg"], key="file_uploader_filter")

    if uploaded_image_filter:
        # Use image hash or bytes to check if it's the same image
        image_bytes_filter = uploaded_image_filter.getvalue()
        image_hash_filter = hash(image_bytes_filter)

        # Store original image and currently displayed image in session state
        if "original_filter_image" not in st.session_state or \
           st.session_state["original_filter_image"]["hash"] != image_hash_filter:
            
            original_image = Image.open(io.BytesIO(image_bytes_filter))
            st.session_state["original_filter_image"] = {
                 "image": original_image,
                 "hash": image_hash_filter,
                 "name": uploaded_image_filter.name
            }
            st.session_state["current_filtered_image"] = original_image # Start with original
            st.session_state["chat_history"].append(("You", f"Uploaded image: {uploaded_image_filter.name} for filtering."))
            st.rerun() # Rerun to set session state

        # Display the current image (either original or filtered)
        if "current_filtered_image" in st.session_state:
             st.image(st.session_state["current_filtered_image"], caption="Processed Image", use_container_width=True)

        filter_options = ["None", "BLUR", "CONTOUR", "DETAIL", "EDGE_ENHANCE", "SMOOTH", "SHARPEN"]
        filter_option = st.selectbox("Choose a filter", filter_options, key="filter_selectbox")

        if st.button("Apply Filter", key="apply_filter_button"):
             if "original_filter_image" in st.session_state:
                original_image = st.session_state["original_filter_image"]["image"]
                with st.spinner(f"Applying {filter_option} filter..."):
                    if filter_option == "None":
                        filtered_image = original_image
                    else:
                        filtered_image = apply_filter(original_image, filter_option)

                    st.session_state["current_filtered_image"] = filtered_image
                    st.session_state["chat_history"].append(("S.A.N.A", f"Applied '{filter_option}' filter to the image."))
                    # No rerun needed immediately after applying filter as session state is updated
                    st.success(f"{filter_option} filter applied!")
             else:
                  st.warning("Please upload an image first.")

# About section
if feature == "About":
    st.header("ℹ️ About Projekt S.A.N.A")
    st.write("This application, **S**ecure **A**utonomous and **N**on-Intrusive **A**ssistant, is a project developed as a demonstration of integrating various AI and information retrieval tools into a single interface.")
    st.write("It leverages APIs from Google Gemini (for chat, summarization, and image description), Wikipedia, and Wolfram Alpha for information and computation, and Hugging Face for image generation and potentially other tasks.")
    st.write("Features include:")
    st.write("- **General Chat:** Engage in conversation powered by Google Gemini.")
    st.write("- **Wikipedia Search:** Get quick summaries from Wikipedia.")
    st.write("- **Wolfram Alpha Queries:** Access computational knowledge.")
    st.write("- **PDF/TXT Summary:** Summarize text documents.")
    st.write("- **Image Description:** Understand the content of images.")
    st.write("- **Image Generation:** Create images from text prompts.")
    st.write("- **Image OCR:** Extract text from images.")
    st.write("- **Image Filtering:** Apply visual effects to images.")

    st.write("---")
    st.write("This project was developed by:")
    st.write("- Ayan Gantayat")
    st.write("- Anish Bhattacharya")
    st.write("- Shaurya Bhandarkar")
    st.write("- Kriday Moudgil")
    st.write("- Darshan Saravanan")
    st.write("Thank you for using S.A.N.A!")
