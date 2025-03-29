import streamlit as st
from streamlit.components.v1 import html
import datetime
import uuid
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
    GENAI_API_KEY =  "AIzaSyABzyFOf6p7izi7VCWIb_Ypf-vZikqlh7o"
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
    APP_ID =  "PHP8VP-Y7P8Y25TTW"
    wolfram_client = wolframalpha.Client(APP_ID)
except KeyError:
    st.error("Error: APP_ID not found in Streamlit secrets.")
    wolfram_client = None

HF_API_KEY = "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"
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
#          Chat History             #
#####################################

def initialize_chat_state():
    """Initialize the chat state if it doesn't exist"""
    if "chats" not in st.session_state:
        st.session_state["chats"] = {
            "default": {
                "id": "default",
                "name": "New Chat",
                "messages": [],
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "context": ""
            }
        }
    
    if "active_chat_id" not in st.session_state:
        st.session_state["active_chat_id"] = "default"
        
    if "feature" not in st.session_state:
        st.session_state["feature"] = "General Chat"
        
    if "pdf" not in st.session_state:
        st.session_state["pdf"] = ""

def get_active_chat():
    """Get the active chat dictionary"""
    return st.session_state["chats"][st.session_state["active_chat_id"]]

def create_new_chat(name="New Chat"):
    """Create a new chat and set it as active"""
    chat_id = str(uuid.uuid4())
    st.session_state["chats"][chat_id] = {
        "id": chat_id,
        "name": name,
        "messages": [],
        "created_at": datetime.datetime.now().isoformat(),
        "updated_at": datetime.datetime.now().isoformat(),
        "context": ""
    }
    st.session_state["active_chat_id"] = chat_id
    return chat_id

def delete_chat(chat_id):
    """Delete a chat by ID"""
    if chat_id in st.session_state["chats"]:
        del st.session_state["chats"][chat_id]
        # If we deleted the active chat, set a new active chat
        if st.session_state["active_chat_id"] == chat_id:
            if st.session_state["chats"]:
                st.session_state["active_chat_id"] = next(iter(st.session_state["chats"]))
            else:
                create_new_chat()

def rename_chat(chat_id, new_name):
    """Rename a chat by ID"""
    if chat_id in st.session_state["chats"] and new_name.strip():
        st.session_state["chats"][chat_id]["name"] = new_name
        st.session_state["chats"][chat_id]["updated_at"] = datetime.datetime.now().isoformat()

def add_message(chat_id, role, content, metadata=None):
    """Add a message to a chat"""
    if chat_id in st.session_state["chats"]:
        if metadata is None:
            metadata = {}
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat(),
            "metadata": metadata
        }
        
        st.session_state["chats"][chat_id]["messages"].append(message)
        st.session_state["chats"][chat_id]["updated_at"] = datetime.datetime.now().isoformat()
        
        # Update context for AI
        if role == "user":
            st.session_state["chats"][chat_id]["context"] += f"User: {content}\n"
        elif role == "assistant":
            st.session_state["chats"][chat_id]["context"] += f"Assistant: {content}\n"

def clear_chat(chat_id):
    """Clear all messages in a chat"""
    if chat_id in st.session_state["chats"]:
        st.session_state["chats"][chat_id]["messages"] = []
        st.session_state["chats"][chat_id]["context"] = ""
        st.session_state["chats"][chat_id]["updated_at"] = datetime.datetime.now().isoformat()

def export_chat(chat_id):
    """Export chat as text"""
    if chat_id in st.session_state["chats"]:
        chat = st.session_state["chats"][chat_id]
        export_text = f"# {chat['name']}\n"
        export_text += f"Created: {chat['created_at']}\n\n"
        
        for msg in chat["messages"]:
            timestamp = datetime.datetime.fromisoformat(msg["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            export_text += f"[{timestamp}] {msg['role'].capitalize()}: {msg['content']}\n\n"
        
        return export_text
    return ""

#####################################
#          Streamlit UI             #
#####################################

def main():
    st.set_page_config(
        page_title="Projekt S.A.N.A",
        page_icon=logo,
        layout="wide"
    )
    
    st.markdown(
    """
    <style>
    /* Remove background and add grey outline to all input fields */
    div[data-baseweb="input"] > div {
        background-color: black ;
        border: 1px solid grey ;
        border-radius: 5px ;
        color: black ;
    }
    /* Make the response text white and background black */
    .response-text {
        color: white !important;
        background-color: black !important;
    }

    /* Optional: Add hover effect for better UX */
    div[data-baseweb="input"] > div:hover {
        border-color: #888 !important;
    }

    /* Optional: Add focus effect */
    div[data-baseweb="input"] > div:focus-within {
        border-color: #555 !important;
        outline: none !important;
    }
    }

    </style>
    """,
    unsafe_allow_html=True
)
    
    # Initialize chat state
    initialize_chat_state()
    
    # Header
    st.markdown(
        f"""
        <div style='display: flex; align-items: center;'>
            <img src='{logo}' width='50' style='margin-right: 15px;'>
            <h1 style='margin: 0;'>Projekt S.A.N.A</h1>
        </div>
        """, unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.title("S.A.N.A Settings")
        
        # Feature selection
        feature = st.selectbox("Select a feature:", [
            "General Chat", 
            "Wikipedia Search", 
            "Wolfram Alpha Queries", 
            "PDF/TXT Summary", 
            "Image Description", 
            "Image Generation",
            "Image OCR",
            "Image Filtering"
        ], key="feature_select")
        
        st.session_state["feature"] = feature
        
        st.markdown("---")
        
        # Chat management section
        st.subheader("💬 Chat Management")
        
        # Create new chat
        new_chat_name = st.text_input("New chat name:", placeholder="Enter chat name...")
        if st.button("➕ Create New Chat"):
            if new_chat_name:
                create_new_chat(new_chat_name)
                st.rerun()
            else:
                create_new_chat()
                st.rerun()
        
        # List of chats
        st.subheader("Your Chats")
        
        # Sort chats by updated_at (most recent first)
        sorted_chats = sorted(
            st.session_state["chats"].values(),
            key=lambda x: x["updated_at"],
            reverse=True
        )
        
        for chat in sorted_chats:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            # Chat name with timestamp
            chat_date = datetime.datetime.fromisoformat(chat["updated_at"]).strftime("%m/%d %H:%M")
            
            # Highlight active chat
            if chat["id"] == st.session_state["active_chat_id"]:
                chat_name_display = f"**→ {chat['name']}** ({chat_date})"
            else:
                chat_name_display = f"{chat['name']} ({chat_date})"
            
            if col1.button(chat_name_display, key=f"select_{chat['id']}"):
                st.session_state["active_chat_id"] = chat["id"]
                st.rerun()
            
            # Edit button
            if col2.button("✏️", key=f"edit_{chat['id']}"):
                st.session_state[f"editing_{chat['id']}"] = True
            
            # Delete button
            if col3.button("🗑️", key=f"delete_{chat['id']}"):
                delete_chat(chat["id"])
                st.rerun()
            
            # Edit form
            if st.session_state.get(f"editing_{chat['id']}", False):
                with st.form(key=f"rename_form_{chat['id']}"):
                    new_name = st.text_input("New name:", value=chat["name"])
                    col1, col2 = st.columns(2)
                    if col1.form_submit_button("Save"):
                        rename_chat(chat["id"], new_name)
                        st.session_state[f"editing_{chat['id']}"] = False
                        st.rerun()
                    if col2.form_submit_button("Cancel"):
                        st.session_state[f"editing_{chat['id']}"] = False
                        st.rerun()
        
        st.markdown("---")
        
        # Chat actions
        st.subheader("Chat Actions")
        col1, col2 = st.columns(2)
        
        if col1.button("🗑️ Clear Current Chat"):
            clear_chat(st.session_state["active_chat_id"])
            st.rerun()
        
        if col2.button("📤 Export Chat"):
            chat_text = export_chat(st.session_state["active_chat_id"])
            st.download_button(
                label="Download Chat",
                data=chat_text,
                file_name=f"sana_chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

    # Main content area
    active_chat = get_active_chat()
    
    # Chat header
    st.subheader(f"Chat: {active_chat['name']}")
    
    # Chat messages container with custom styling
    chat_container = st.container()
    
    with chat_container:
        # Display messages with better styling
        for message in active_chat["messages"]:
            timestamp = datetime.datetime.fromisoformat(message["timestamp"]).strftime("%H:%M:%S")
            
            if message["role"] == "user":
                st.markdown(
                    f"""
                    <div style='background-color: #1d2224; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                        <div style='display: flex; justify-content: space-between;'>
                            <strong>You</strong>
                            <span style='color: #888; font-size: 0.8em;'>{timestamp}</span>
                        </div>
                        <div style='margin-top: 5px;'>{message['content']}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:  # assistant
                st.markdown(
                    f"""
                    <div style='background-color: #1d2224; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                        <div style='display: flex; justify-content: space-between;'>
                            <strong>S.A.N.A</strong>
                            <span style='color: #888; font-size: 0.8em;'>{timestamp}</span>
                        </div>
                        <div style='margin-top: 5px;'>{message['content']}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Display any metadata (like images) if present
                if "image_path" in message.get("metadata", {}):
                    st.image(message["metadata"]["image_path"])
    
    # Feature-specific UI
    st.markdown("---")
    
    # Display the input field for relevant features
    if st.session_state["feature"] in ["General Chat", "Wikipedia Search", "Wolfram Alpha Queries"]:
        with st.form("InputForm"):
            user_input = st.text_input("💬 Type your query:", placeholder="Ask anything...", key="user_input")
            
            if st.form_submit_button("Send"):
                if user_input:
                    # Add user message
                    add_message(st.session_state["active_chat_id"], "user", user_input)
                    
                    # Generate response based on feature
                    if st.session_state["feature"] == "Wikipedia Search":
                        response = search_wikipedia(user_input)
                    elif st.session_state["feature"] == "Wolfram Alpha Queries":
                        response = query_wolfram_alpha(user_input)
                    else:  # General Chat
                        response = query_google_gemini(user_input, active_chat["context"])
                    
                    # Add assistant message
                    add_message(st.session_state["active_chat_id"], "assistant", response)
                    
                    st.rerun()

    # PDF/TXT Summary
    if st.session_state["feature"] == "PDF/TXT Summary":
        uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
        if uploaded_file:
            if uploaded_file != st.session_state["pdf"]:
                # Add user message about the upload
                add_message(
                    st.session_state["active_chat_id"], 
                    "user", 
                    f"I've uploaded a file: {uploaded_file.name}"
                )
                
                st.success("File uploaded successfully!")
                
                # Process the file
                with st.spinner("Processing file..."):
                    summary = process_uploaded_file(uploaded_file)
                
                # Add assistant message with the summary
                add_message(
                    st.session_state["active_chat_id"], 
                    "assistant", 
                    f"Here's a summary of {uploaded_file.name}:\n\n{summary}"
                )
                
                st.session_state["pdf"] = uploaded_file
                st.rerun()
            
            # Display the summary
            st.markdown(f"**📜 Summary of {uploaded_file.name}:**")
            st.write(active_chat["messages"][-1]["content"])

    # Image Description
    if st.session_state["feature"] == "Image Description":
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Get description
                with st.spinner("Analyzing image..."):
                    description = describe_image(image)
                
                # Add messages to chat
                add_message(
                    st.session_state["active_chat_id"], 
                    "user", 
                    f"I've uploaded an image: {uploaded_image.name}"
                )
                
                add_message(
                    st.session_state["active_chat_id"], 
                    "assistant", 
                    f"Here's what I see in the image:\n\n{description}",
                    metadata={"image_type": "uploaded"}
                )
                
                st.markdown(f"**🖼️ Description:** {description}")
        
        with col2:
            captured_image = st.camera_input("Take a picture")
            if captured_image:
                image = Image.open(captured_image)
                
                # Get description
                with st.spinner("Analyzing image..."):
                    description = describe_image(image)
                
                # Add messages to chat
                add_message(
                    st.session_state["active_chat_id"], 
                    "user", 
                    "I've taken a picture with the camera"
                )
                
                add_message(
                    st.session_state["active_chat_id"], 
                    "assistant", 
                    f"Here's what I see in the image:\n\n{description}",
                    metadata={"image_type": "captured"}
                )
                
                st.markdown(f"**🖼️ Description:** {description}")

    # Image Generation
    if st.session_state["feature"] == "Image Generation":
        prompt = st.text_input("🎨 Enter a prompt for the AI-generated image:")
        if st.button("Generate Image"):
            if prompt:
                # Add user message
                add_message(
                    st.session_state["active_chat_id"], 
                    "user", 
                    f"Generate an image of: {prompt}"
                )
                
                with st.spinner("Generating image..."):
                    generated_img = generate_image(prompt)
                    
                    if generated_img:
                        # Save image to a BytesIO object to display
                        img_byte_arr = io.BytesIO()
                        generated_img.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        
                        # Add assistant message
                        add_message(
                            st.session_state["active_chat_id"], 
                            "assistant", 
                            f"I've generated an image based on your prompt: '{prompt}'",
                            metadata={"image_type": "generated", "prompt": prompt}
                        )
                        
                        st.image(generated_img, caption="🖼️ AI-Generated Image", use_container_width=True)
                    else:
                        # Add error message
                        add_message(
                            st.session_state["active_chat_id"], 
                            "assistant", 
                            f"I'm sorry, I couldn't generate an image for: '{prompt}'. Please try a different prompt."
                        )
                        
                        st.error("Failed to generate image. Try a different prompt.")

    # Enhanced Image OCR
    if st.session_state["feature"] == "Image OCR":
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Selected Image", use_container_width=True)
                
                # Extract text
                with st.spinner("Extracting text..."):
                    extracted_text = image_ocr(image)
                
                # Add messages to chat
                add_message(
                    st.session_state["active_chat_id"], 
                    "user", 
                    f"Extract text from this image: {uploaded_image.name}"
                )
                
                add_message(
                    st.session_state["active_chat_id"], 
                    "assistant", 
                    f"Here's the text I extracted from the image:\n\n{extracted_text}"
                )
                
                st.text_area("Extracted Text", extracted_text, height=150)
        
        with col2:
            camera_image = st.camera_input("Capture an image")
            if camera_image:
                image = Image.open(camera_image)
                
                # Extract text
                with st.spinner("Extracting text..."):
                    extracted_text = image_ocr(image)
                
                # Add messages to chat
                add_message(
                    st.session_state["active_chat_id"], 
                    "user", 
                    "Extract text from this camera image"
                )
                
                add_message(
                    st.session_state["active_chat_id"], 
                    "assistant", 
                    f"Here's the text I extracted from the camera image:\n\n{extracted_text}"
                )
                
                st.text_area("Extracted Text from Camera", extracted_text, height=150)

    # Enhanced Image Filtering
    if st.session_state["feature"] == "Image Filtering":
        uploaded_image = st.file_uploader("Upload an image to apply filters", type=["jpg", "png", "jpeg"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_container_width=True)
            
            filter_option = st.selectbox("Choose a filter", ["None", "BLUR", "CONTOUR", "DETAIL"])
            
            if st.button("Apply Filter"):
                # Add user message
                add_message(
                    st.session_state["active_chat_id"], 
                    "user", 
                    f"Apply {filter_option} filter to image: {uploaded_image.name}"
                )
                
                if filter_option != "None":
                    filtered_image = apply_filter(image, filter_option)
                    
                    # Add assistant message
                    add_message(
                        st.session_state["active_chat_id"], 
                        "assistant", 
                        f"I've applied the {filter_option} filter to your image."
                    )
                    
                    st.image(filtered_image, caption=f"Filtered Image ({filter_option})", use_container_width=True)
                else:
                    # Add assistant message
                    add_message(
                        st.session_state["active_chat_id"], 
                        "assistant", 
                        "No filter was applied to the image."
                    )
                    
                    st.image(image, caption="No Filter Applied", use_container_width=True)

if __name__ == "__main__":
    main()

