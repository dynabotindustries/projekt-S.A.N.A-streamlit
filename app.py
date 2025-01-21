import streamlit as st
import wikipedia
import wolframalpha
import google.generativeai as genai

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
    model_name="gemini-2.0-flash-exp",    # Defines Gemini model to be used
    system_instruction=[system_prompt]    # Sets system instruction to be followed as per variable `system_prompt`
)

# WolframAlpha App ID
APP_ID = st.secrets["APP_ID"]  # Replace with your actual API key

# APP logo
logo = "https://avatars.githubusercontent.com/u/175069629?v=4"

## Functions for the assistant

# Function to search through Wikipedia
def search_wikipedia(query):
    try:
        # return a summary of all content found on Wikipedia if the query successfully parses information
        result = wikipedia.summary(query, sentences=2)
        return result
    except wikipedia.exceptions.DisambiguationError as e:
        # return an error if prompt is ambiguous
        return "Multiple meanings detected. Please specify: " + ", ".join(e.options[:5])
    except wikipedia.exceptions.PageError:
        # return an error if no matching results are found
        return "No results found on Wikipedia."

# Function to query WolframAlpha
def query_wolfram_alpha(query):
    # Initialize the client
    client = wolframalpha.Client(APP_ID)
    try:
        # return the result upon a successful query
        res = client.query(query)
        return next(res.results).text
    except Exception:
        # return an error upon any exception
        return "No results found on Wolfram Alpha."

# Function to query Gemini
def query_google_gemini(query, context):
    try:
        # Combine context with the current query
        conversation_input = context + f"\nUser: {query}\nAssistant:"
        # Generate a response using the specified Gemini Model
        response = model.generate_content(conversation_input)
        # return the generated response
        return response.text
    except Exception as e:
        # return an error upon any exception
        return f"An error occurred while fetching from Google Gemini: {str(e)}"

# Streamlit App
st.set_page_config(page_title="Projekt S.A.N.A", page_icon=logo, layout="wide")

# Sidebar
with st.sidebar:
    st.title("S.A.N.A Settings")
    st.markdown("⚙️ **Customize your assistant experience (Coming Soon!)**")
    st.markdown("---")
    st.markdown("Use the features below to interact with S.A.N.A:")
    st.markdown("1. Wikipedia Search\n2. Wolfram Alpha Queries\n3. Google Gemini Chat(No Conversations Recorded by any Party Concerned)")

# Main App

# Logo and Title in HTML format for inline logo
st.markdown(f"<h1><img src='{logo}' width=70 style='display:inline-block; margin-right:15px'></img><b>Projekt S.A.N.A:</b></h1>", unsafe_allow_html=True)

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
    ["General Chat", "Wikipedia Search", "Wolfram Alpha Queries"], index=0)

# User Input Section
user_input = st.text_input("💬 Type your query below:", placeholder="Ask anything...")

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

        # Add response to chat history as `S.A.N.A.`
        st.session_state["chat_history"].append(("S.A.N.A", response))

        # Update context for chat-based features
        st.session_state["context"] += f"User: {user_input}\nAssistant: {response}\n"

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

# Clear History Button
st.write("---")
if st.button("Clear Chat History"):
    st.session_state["chat_history"] = []
    st.session_state["context"] = ""
    st.success("Chat history cleared!")
