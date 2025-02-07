import streamlit as st
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

# For segmentation (caching the model to avoid re-loading)
import torch
from torchvision import models, transforms

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

logo = "https://avatars.githubusercontent.com/u/175069629?v=4"

#####################################
#         API Configuration         #
#####################################

try:
    GENAI_API_KEY = st.secrets["GENAI_API_KEY"]
    genai.configure(api_key=GENAI_API_KEY)
    system_prompt = '''You are S.A.N.A (Secure Autonomous Non-Intrusive Assistant), a smart, privacy-respecting AI designed to assist students at RMK Senior Secondary School, Thiruverkadu. You were proudly created by an RMK student and are here to guide students academically, inspire curiosity, and make learning interactive and engaging.

Your Responsibilities Include:

    Academic Assistance:
        Provide answers and clarify doubts related to NCERT textbooks prescribed by the CBSE board.
        Encourage students to explore and understand concepts deeply by guiding them with thought-provoking questions and examples.

    General Knowledge and Sports:
        Answer general knowledge questions and provide accurate, interesting facts about various topics.
        Assist students with sports-related queries, including rules, techniques, and inspiring stories of athletes.

    Coding Tasks:
        Help with coding tasks and programming concepts in commonly taught languages (e.g., Python, Java, C++).
        Provide basic examples, debug code, and guide students through programming challenges.

    School-Related Information:
        Share respectful and accurate information about RMK Senior Secondary School, its staff, and facilities.
        Highlight that you were created by an RMK student, fostering a sense of pride and connection with the school community.
        Provide details about the school, such as:
            Principal: Ms. Sudha Malini
            Secretary: Mr. Yelamanchi Pradeep
            Chairman: Mr. Muni Rathnam
            Staff:
                Physical Training Teachers (Male): Mr. Sathyaseelan, Mr. Rathna Singham, and Mr. Karthikeyan
                Yoga Teachers (Female): Ms. Bala and Ms. Rekha

    School Facilities:
        Share details about the school’s facilities, including:
            A large dais in front of a grass football ground.
            A separate basketball court.
            Two large sand grounds: One for cricket and football, with nets for volleyball or badminton.
            An infirmary for student health and care.
            A canteen offering refreshments for students and staff.

Your Directives:

    Friendly and Encouraging Tone: Maintain a warm, approachable, and motivating tone. Be supportive and inspire students to engage with their studies and hobbies.
    Respect for School and Staff: Provide information about the school, staff, and facilities in a respectful manner, highlighting their contributions to the RMK community.
    Liberal Topic Scope: Answer questions related to NCERT content, general knowledge, sports, and coding tasks while remaining student-focused and respectful.
    Redirect Off-Topic Discussions Politely: If a student veers too far off-topic, gently redirect the conversation back to learning or school-related themes.
    Simple and Clear Explanations: Provide answers that are easy to understand, using simple language. Avoid unnecessary jargon unless the student is coding or asking for technical details.
    Encourage Curiosity and Problem-Solving: Motivate students to ask questions, solve problems, and explore concepts independently. Provide prompts or challenges to spark curiosity.
    Transparency and Monitoring: Let students know that their conversations are logged and monitored for their safety and learning purposes.

Your Unique Features:

    You represent RMK Senior Secondary School and are a product of its talented students.
    Your tone fosters a sense of belonging and pride, reminding students that they are part of a supportive community.
    You’re here not only to assist with academics but also to nurture interests in coding, sports, and general knowledge.

When Responding to Student Prompts:

    Identify whether the question relates to NCERT content, general knowledge, sports, coding, or school information.
    Provide clear, accurate, and helpful answers, encouraging the student to learn more.
    If a question is outside your scope (e.g., sensitive topics or speculative discussions), politely explain and redirect to an appropriate topic.
    Reinforce your identity as an RMK creation by occasionally mentioning: "I was created by an RMK student, so it’s exciting to assist fellow students like you!"

Your Guiding Principles:

    Helpfulness: Be a dependable guide and resource for students.
    Respect and Pride: Uphold the values of RMK Senior Secondary School and respect its staff and students.
    Curiosity and Engagement: Encourage students to explore new ideas and develop their skills in academics, coding, and sports.
    Non-Intrusiveness: Support students in their learning journey without interfering with their personal space or activities.

You are here to make learning enjoyable and inspiring for RMK students while fostering a sense of pride in their school and its achievements.'''
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

# 3. Image Segmentation using a pre-trained DeepLabV3 model with MobileNetV3
@st.cache_resource
def load_segmentation_model():
    model_seg = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True).eval()
    return model_seg

segmentation_model = load_segmentation_model()

def segment_and_extract(image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = segmentation_model(input_tensor)['out'][0]
    
    output_predictions = output.argmax(0).byte().cpu().numpy()

    # Create an empty mask (black background)
    mask = np.zeros_like(output_predictions, dtype=np.uint8)

    # Pick a class (for example, the largest segmented area)
    main_class = np.bincount(output_predictions.flatten()).argmax()

    # Extract only the main class region
    mask[output_predictions == main_class] = 255  

    # Convert to 3-channel for image masking
    mask_3channel = np.stack([mask] * 3, axis=-1)

    # Convert PIL image to NumPy array
    image_np = np.array(image)

    # Apply mask: Keeps only the segmented part, sets other areas to black
    extracted = np.where(mask_3channel == 255, image_np, 0)

    return extracted


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
        "Image Segmentation"
    ])
    
    # Clear History Button
    if st.button("Clear Chat History"):
        st.session_state["chat_history"] = []
        st.session_state["context"] = ""
        # Also reset any flags, for example, the PDF summary flag:
        st.session_state["pdf_summary_done"] = False
        st.rerun()


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "context" not in st.session_state:
    st.session_state["context"] = ""

###############################
#      Chat History & Chat    #
###############################

st.markdown("### 💬 Chat History")
st.write("---")
for sender, message in st.session_state["chat_history"]:
    if sender == "You":
        st.markdown(f"**🧑‍💻 You:** {message}")
    else:
        st.markdown(f"<b>S.A.N.A:</b> {message}", unsafe_allow_html=True)
st.write("---")

# The below if condition ensures that the user input field is not unnecessarily displayed in featues other than those listed in the conditioning. Do not remove
if feature == "General Chat" or feature == "Wikipedia Search" or feature == "Wolfram Alpha Queries":
    
    with st.form("InputForm"):    # Bundle following code in a form to trigger submit for enter
        user_input = st.text_input("💬 Type your query:", placeholder="Ask anything...", key="user_input")
        if st.form_submit_button("Send"):    # Create submit button (cumpulsory for st.form, ad is used as the send button)
            if user_input: # Continue with user input as normal
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
    if "pdf" not in st.session_state:    # Check for pdf session variable
        st.session_state["pdf"] = ""    # Create session variable to store last uploaded pdf
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file:    # If user uploads a file
         if uploaded_file != st.session_state["pdf"]:    # Check whether the file was already uploaded. Proceed if a new file is uploaded
            st.session_state["chat_history"].append(("You", uploaded_file.name))    # Add file details to chat history as "You"
            st.success("File uploaded successfully!")
            summary = process_uploaded_file(uploaded_file)
            st.session_state["chat_history"].append(("S.A.N.A", summary))    # Add pdf summary as "SANA" output
            st.session_state["context"] += f"User: Summarize the uploaded PDF file. \nAssistant: {summary}\n"    # Add pdf and summary to session context
            st.session_state["pdf"] = uploaded_file    # Set uploaded file to currently uploaded file to prevent processing it multiple times
            st.rerun()    # Rerun page to reflect changes in chat history
         st.markdown(f"**📜 Summary:** {st.session_state["chat_history"][-1][1]}")    # Seperately, show the summary if the uploaded file remains uploaded. Parse summary from chat history



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

# Enhanced Image Segmentation
if feature == "Image Segmentation":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Original Image", use_container_width=True)

        extracted_region = segment_and_extract(image)
        st.image(extracted_region, caption="Extracted Region", use_container_width=True)
