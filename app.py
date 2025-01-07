import streamlit as st
import wikipedia
import wolframalpha
import pyttsx3

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# WolframAlpha App ID
APP_ID = "PHP8VP-Y7P8Y25TTW"  # Replace with your actual API key


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


def speak(text):
    """Use text-to-speech to speak the provided text."""
    engine.say(text)
    engine.runAndWait()


# Streamlit App
st.set_page_config(page_title="Projekt S.A.N.A.", page_icon="🤖", layout="wide")

# Sidebar
st.sidebar.title("S.A.N.A. Settings")
voice_option = st.sidebar.selectbox("Voice", ["Male", "Female"])
st.sidebar.markdown("Change the voice type for S.A.N.A.")

if voice_option == "Male":
    engine.setProperty("voice", engine.getProperty("voices")[0].id)
else:
    engine.setProperty("voice", engine.getProperty("voices")[1].id)

# Main App
st.title("Projekt S.A.N.A.")
st.markdown("**Secure, Autonomous, Non-intrusive Assistant**")

# Display input and chat history
chat_history = st.session_state.get("chat_history", [])
user_input = st.text_input("Your Message:", key="user_input")

if st.button("Send"):
    if user_input:
        # Add user message to chat history
        chat_history.append(("You", user_input))

        # Process the input
        user_input = user_input.strip().lower()
        if any(phrase in user_input for phrase in ["who are you", "what are you", "who is sana", "what is sana"]):
            response = (
                "Hello! I'm S.A.N.A (Secure, Autonomous, Non-intrusive Assistant), "
                "an open-source virtual assistant designed to prioritize your privacy. "
                "I'm here to help you with your tasks and queries without compromising your data."
            )
        elif "search" in user_input:
            query = user_input.replace("search", "").strip()
            response = search_wikipedia(query)
        elif "play" in user_input:
            query = user_input.replace("play", "").strip()
            response = f"Cannot play media in Streamlit mode. Searched: {query}"
        elif "close" in user_input:
            response = "Cannot close tabs in Streamlit mode."
        else:
            response = query_wolfram_alpha(user_input)

        # Add S.A.N.A.'s response to chat history
        chat_history.append(("S.A.N.A.", response))
        speak(response)

# Display chat history
for sender, message in chat_history:
    if sender == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**S.A.N.A.:** {message}")

# Update session state with chat history
st.session_state["chat_history"] = chat_history

# Reset Button
if st.button("Clear History"):
    st.session_state["chat_history"] = []
    st.experimental_rerun()
