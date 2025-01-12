import streamlit as st
import wikipedia
import wolframalpha
import google.generativeai as genai

# Google Gemini API key
GENAI_API_KEY = st.secrets[GENAI_API_KEY]  # Replace with your actual API key
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# WolframAlpha App ID
APP_ID = st.secrets[APP_ID]  # Replace with your actual API key


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
        # Combine context with the current query
        conversation_input = context + f"\nUser: {query}\nAssistant:"
        response = model.generate_content(conversation_input)
        return response.text
    except Exception as e:
        return f"An error occurred while fetching from Google Gemini: {str(e)}"


# Streamlit App
st.set_page_config(page_title="Projekt S.A.N.A.", page_icon="🤖", layout="wide")

# Sidebar
st.sidebar.title("S.A.N.A. Settings")
st.sidebar.markdown("Modify settings for S.A.N.A.")

# Main App
st.title("Projekt S.A.N.A.")
st.markdown("**Secure, Autonomous, Non-intrusive Assistant**")

# Initialize session state for chat history and context
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "context" not in st.session_state:
    st.session_state["context"] = ""

# Input and chat history
user_input = st.text_input("Your Message:", key="user_input")

if st.button("Send"):
    if user_input:
        # Add user message to chat history
        st.session_state["chat_history"].append(("You", user_input))

        # Process the input
        user_input_lower = user_input.strip().lower()
        if any(phrase in user_input_lower for phrase in ["who are you", "what are you", "who is sana", "what is sana"]):
            response = (
                "Hello! I'm S.A.N.A (Secure, Autonomous, Non-intrusive Assistant), "
                "an open-source virtual assistant designed to prioritize your privacy. "
                "I'm here to help you with your tasks and queries without compromising your data."
            )
        elif "search" in user_input_lower:
            query = user_input_lower.replace("search", "").strip()
            response = search_wikipedia(query)
        elif "wolfram" in user_input_lower:
            query = user_input_lower.replace("wolfram", "").strip()
            response = query_wolfram_alpha(query)
        else:
            # Use conversation history as context for Google Gemini
            response = query_google_gemini(user_input, st.session_state["context"])

        # Add response to chat history
        st.session_state["chat_history"].append(("S.A.N.A.", response))

        # Update context for future queries
        st.session_state["context"] += f"User: {user_input}\nAssistant: {response}\n"

# Display chat history
for sender, message in st.session_state["chat_history"]:
    if sender == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**S.A.N.A.:** {message}")

# Clear history button
if st.button("Clear History"):
    st.session_state["chat_history"] = []
    st.session_state["context"] = ""
    st.experimental_rerun()
