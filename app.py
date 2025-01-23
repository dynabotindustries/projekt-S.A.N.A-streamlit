import streamlit as st
import wikipedia
import wolframalpha
import google.generativeai as genai
from PyPDF2 import PdfReader

# Google Gemini API key
GENAI_API_KEY = st.secrets["GENAI_API_KEY"]  # Replace with your actual API key
genai.configure(api_key=GENAI_API_KEY)

system_prompt = '''You are S.A.N.A (Secure Autonomous Non-Intrusive Assistant), a privacy-respecting chatbot designed to engage in constructive and meaningful conversations. Your goal is to answer user queries effectively or chat on their preferred topics while ensuring the interaction stays engaging and enjoyable.

    Do not discuss sensitive or controversial topics.
    If such topics arise, politely redirect the conversation to a positive or neutral subject.
    Keep responses friendly, engaging, and concise.
    Use emojis occasionally to make the conversation lively and relatable.

Always prioritize user satisfaction while respecting privacy and maintaining a helpful tone.'''

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    system_instruction=[system_prompt]
)

# WolframAlpha App ID
APP_ID = st.secrets["APP_ID"]  # Replace with your actual API key

# APP logo
logo = "https://avatars.githubusercontent.com/u/175069629?v=4"

# Functions for the assistant

def search_wikipedia(query):
    try:
        result = wikipedia.summary(query, sentences=2)
        return result
    except wikipedia.exceptions.DisambiguationError as e:
        return "Multiple meanings detected. Please specify: " + ", ".join(e.options[:5])
    except wikipedia.exceptions.PageError:
        return "No results found on Wikipedia."

def query_wolfram_alpha(query):
    client = wolframalpha.Client(APP_ID)
    try:
        res = client.query(query)
        return next(res.results).text
    except Exception:
        return "No results found on Wolfram Alpha."

def query_google_gemini(query, context):
    try:
        conversation_input = context + f"\nUser: {query}\nAssistant:"
        response = model.generate_content(conversation_input)
        return response.text
    except Exception as e:
        return f"An error occurred while fetching from Google Gemini: {str(e)}"

def summarize_file_with_gemini(file_content):
    try:
        response = model.generate_content(file_content)
        return response.text
    except Exception as e:
        return f"An error occurred while generating summary with Gemini: {str(e)}"

# Streamlit App
st.set_page_config(page_title="Projekt S.A.N.A", page_icon=logo, layout="wide")

# Sidebar
with st.sidebar:
    st.title("S.A.N.A Settings")
    st.markdown("⚙️ **Customize your assistant experience (Coming Soon!)**")
    st.markdown("---")
    st.markdown("Use the features below to interact with S.A.N.A:")
    st.markdown("1. Wikipedia Search\n2. Wolfram Alpha Queries\n3. Google Gemini Chat\n4. File Upload for Text Summarization")

# Main App
st.markdown(f"<h1><img src='{logo}' width=70 style='display:inline-block; margin-right:15px'></img><b>Projekt S.A.N.A:</b></h1>", unsafe_allow_html=True)
st.markdown("""
**S.A.N.A** is a secure, autonomous, and non-intrusive virtual assistant. 
Feel free to ask me anything! 😊
""")
st.markdown("---")

# Initialize session variables
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []   # Initialize chat history
if "context" not in st.session_state:
    st.session_state["context"] = ""   # Initialize context

# Feature Selection Dropdown
feature = st.selectbox("Select a feature to use:", 
    ["General Chat", "Wikipedia Search", "Wolfram Alpha Queries", "File Upload"], index=0)

# Display Chat History
st.markdown("### 💬 Chat History")
st.write("---")
for sender, message in st.session_state["chat_history"]:   # Parse session chat history tuple as (sender, message)
    if sender == "You":
        # Render user prompt
        st.markdown(f"**🧑‍💻 You:** {message}")
    elif sender == "S.A.N.A":
        # Render logo and the response inline
        st.markdown(f"<img src='{logo}' width=20 style='display:inline-block; margin-right:10px'></img><b>S.A.N.A:</b> {message}", unsafe_allow_html=True)
    else:
        st.markdown(f"**❗Unknown Sender:** {message}")

# File upload section for PDF/TXT
if feature == "File Upload":
    uploaded_file = st.file_uploader("Upload a PDF/TXT file for summarization", type=["pdf", "txt"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            # Extract text from PDF and send it to Gemini for summarization
            pdf_reader = PdfReader(uploaded_file)
            file_content = ""
            for page in pdf_reader.pages:
                file_content += page.extract_text()
            summary = summarize_file_with_gemini(file_content)
        elif uploaded_file.type == "text/plain":
            # Read text file and send content to Gemini for summarization
            file_content = uploaded_file.read().decode("utf-8")
            summary = summarize_file_with_gemini(file_content)
        
        st.markdown("### 📄 File Summary:")
        st.write(summary)
        
        # Set the response from the file summary
        response = summary
    else:
        response = ""  # Default empty response

# User Input Section (Text Input Box)
user_input = st.text_input("💬 Type your query below:", placeholder="Ask anything...", key="user_input")

# Create a button to submit user input
if st.button("Send") or user_input:  # If "Send" button is pressed or user input is not empty
    if user_input:
        # Add user message to chat history as `You`
        st.session_state["chat_history"].append(("You", user_input))

        # Process based on selected feature
        if feature == "Wikipedia Search":
            response = search_wikipedia(user_input)
        elif feature == "Wolfram Alpha Queries":
            response = query_wolfram_alpha(user_input)
        elif feature == "General Chat":
            response = query_google_gemini(user_input, st.session_state["context"])

        # Add response to chat history as `S.A.N.A.`
        st.session_state["chat_history"].append(("S.A.N.A", response))

        # Update context for chat-based features
        st.session_state["context"] += f"User: {user_input}\nAssistant: {response}\n"
