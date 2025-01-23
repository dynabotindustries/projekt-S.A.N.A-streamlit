import streamlit as st
import wikipedia
import wolframalpha
import google.generativeai as genai
import PyPDF2

# Google Gemini API key
GENAI_API_KEY = st.secrets["GENAI_API_KEY"]  # Replace with your actual API key
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
            A canteen offering refreshments for students and staff.'''
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",    # Defines Gemini model to be used
    system_instruction=[system_prompt]    # Sets system instruction to be followed as per variable `system_prompt`
)

# WolframAlpha App ID
APP_ID = st.secrets["APP_ID"]  # Replace with your actual API key

# APP logo
logo = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fsulcdn.azureedge.net%2Fbiz-live%2Fimg%2F452578-2712466-28022017141422.jpeg&f=1&nofb=1&ipt=42a20b04f760c91a996be135607e412eca2a1b29d3b555dd27fbb8473916f93b&ipo=images"

## Functions for the assistant

# Function to search through Wikipedia
def search_wikipedia(query):
    try:
        result = wikipedia.summary(query, sentences=2)
        return result
    except wikipedia.exceptions.DisambiguationError as e:
        return "Multiple meanings detected. Please specify: " + ", ".join(e.options[:5])
    except wikipedia.exceptions.PageError:
        return "No results found on Wikipedia."

# Function to query WolframAlpha
def query_wolfram_alpha(query):
    client = wolframalpha.Client(APP_ID)
    try:
        res = client.query(query)
        return next(res.results).text
    except Exception:
        return "No results found on Wolfram Alpha."

# Function to query Gemini
def query_google_gemini(query, context):
    try:
        conversation_input = context + f"\nUser: {query}\nAssistant:"
        response = model.generate_content(conversation_input)
        return response.text
    except Exception as e:
        return f"An error occurred while fetching from Google Gemini: {str(e)}"

# Function to extract text from PDF
def extract_pdf_summary(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text[:1000]  # Return the first 1000 characters for preview
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# Streamlit App
st.set_page_config(page_title="Projekt S.A.N.A for RMK School", page_icon=logo, layout="wide")

# Sidebar
with st.sidebar:
    st.title("S.A.N.A Settings")
    st.markdown("⚙️ **Customize your assistant experience (coming soon!)**")
    st.markdown("---")
    st.markdown("Use the features below to interact with S.A.N.A:")
    st.markdown("1. Wikipedia Search\n2. Wolfram Alpha Queries\n3. Google Gemini Chat\n4. PDF Summary")

# Main App

# Logo and Title
st.markdown(f"<h1><img src='{logo}' width=70 style='display:inline-block; margin-right:15px'></img><b>Projekt S.A.N.A for RMK School:</b></h1>", unsafe_allow_html=True)

# Add description
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
    ["General Chat", "Wikipedia Search", "Wolfram Alpha Queries", "PDF Summary"], index=0)

# User Input Section
user_input = st.text_input("💬 Type your query below:", placeholder="Ask anything...")

# Handle File Upload for PDF Summary
pdf_file = st.file_uploader("Upload PDF for Summary", type="pdf")

if st.button("Send"):
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
        elif feature == "PDF Summary" and pdf_file:
            response = extract_pdf_summary(pdf_file)

        # Add response to chat history as `S.A.N.A.`
        st.session_state["chat_history"].append(("S.A.N.A", response))

        # Update context for chat-based features
        st.session_state["context"] += f"User: {user_input}\nAssistant: {response}\n"

# Display Chat History
st.markdown("### 💬 Chat History")
st.write("---")
for sender, message in st.session_state["chat_history"]:
    if sender == "You":
        # Render user prompt
        st.markdown(f"**🧑‍💻 You:** {message}")
    elif sender == "S.A.N.A":
        # Render logo and the response inline
        st.markdown(f"<img src='{logo}' width=20 style='display:inline-block; margin-right:10px'></img><b>S.A.N.A:</b> {message}", unsafe_allow_html=True)

# Clear History Button
st.write("---")
if st.button("Clear Chat History"):
    st.session_state["chat_history"] = []
    st.session_state["context"] = ""
    st.success("Chat history cleared!")
