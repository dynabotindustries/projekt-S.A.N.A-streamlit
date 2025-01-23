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

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "context" not in st.session_state:
    st.session_state["context"] = ""

# Feature Selection Dropdown
feature = st.selectbox("Select a feature to use:", 
    ["General Chat", "Wikipedia Search", "Wolfram Alpha Queries", "File Upload (Summarization)"], index=0)

# Feature-Specific Sections
if feature == "File Upload (Summarization)":
    uploaded_file = st.file_uploader("Upload a text or PDF file for summarization:", type=["txt", "pdf"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            text = " ".join(page.extract_text() for page in reader.pages)
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")

        if len(text) > 3000:
            text = text[:3000]  # Truncate text to 3000 characters for processing

        response = query_google_gemini(f"Summarize this: {text}", st.session_state["context"])
        st.markdown(f"### Summary:\n{response}")

# Chat History Section
st.markdown("### 💬 Chat History")
for sender, message in st.session_state["chat_history"]:
    if sender == "You":
        st.markdown(f"**🧑‍💻 You:** {message}")
    elif sender == "S.A.N.A":
        st.markdown(f"<img src='{logo}' width=20 style='display:inline-block; margin-right:10px'></img><b>S.A.N.A:</b> {message}", unsafe_allow_html=True)

# User Input Section at the Bottom
st.markdown("---")
user_input = st.text_input("💬 Type your query below:", placeholder="Ask anything...", key="user_input")
if user_input and st.session_state.get("enter_pressed", False):
    st.session_state["chat_history"].append(("You", user_input))
    if feature == "Wikipedia Search":
        response = search_wikipedia(user_input)
    elif feature == "Wolfram Alpha Queries":
        response = query_wolfram_alpha(user_input)
    elif feature == "General Chat":
        response = query_google_gemini(user_input, st.session_state["context"])
    st.session_state["chat_history"].append(("S.A.N.A", response))
    st.session_state["context"] += f"User: {user_input}\nAssistant: {response}\n"
    st.session_state["user_input"] = ""
    st.session_state["enter_pressed"] = False

if st.button("Send"):
    if user_input:
        st.session_state["enter_pressed"] = True

# Add JavaScript for Enter Key
st.markdown(
    """<script>
    const textInput = document.querySelector("input[data-testid='stTextInput']");
    textInput.addEventListener("keydown", function(event) {
        if (event.key === "Enter") {
            textInput.dispatchEvent(new Event("change", { 'bubbles': true }));
        }
    });
    </script>""",
    unsafe_allow_html=True
)

# Clear History Button
st.write("---")
if st.button("Clear Chat History"):
    st.session_state["chat_history"] = []
    st.session_state["context"] = ""
    st.success("Chat history cleared!")
