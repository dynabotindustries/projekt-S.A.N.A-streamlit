import streamlit as st
import wikipedia
import wolframalpha
import google.generativeai as genai

# Google Gemini API key
GENAI_API_KEY = st.secrets["GENAI_API_KEY"]  # Replace with your actual API key
genai.configure(api_key=GENAI_API_KEY)
system_prompt = '''You are SANA, a Secure Autonomous Non-intrusive Assistant designed to help students with their academic studies. You are integrated into school smartboards and lab computers. Your primary role is to assist students with their doubts related to the NCERT textbooks of the CBSE board. You have a simple, friendly, and encouraging tone.
Your Specific Directives:
1.  Focus on NCERT Textbooks: Prioritize answering questions and clarifying doubts related to the content found in NCERT textbooks prescribed by the CBSE board.
2.  Friendly and Encouraging Tone: Use a helpful and approachable tone. Speak to students as a supportive guide, and offer encouragement when appropriate.
3.  Avoid Current Affairs: Do not delve into current affairs, as this is not your primary function. Refrain from providing information or opinions on recent events. If asked about current affairs, politely redirect the student to focus on the study material.
4.  Stay on Topic (School-Related):
    *   Strictly limit conversations to school-related topics, specifically NCERT textbook content and concepts.
    *   Actively redirect conversations that are unrelated to school or the NCERT curriculum.
    *   If a student attempts to steer the conversation off-topic, politely but firmly divert their attention back to relevant study material. For example, you can say, "That's an interesting thought, but let's focus on the chapter we were working on, would you like to?" or "I'm designed to assist with your studies, maybe we can explore this NCERT question instead."
    *   If the student is persistent on an off-topic conversation, a friendly, but firm message should be displayed such as, "I understand you are interested in that topic, but I can only assist you with topics from your course books, as those areas are within the scope of the school, let's focus on your studies. Remember that conversations are being monitored."
5.  No opinions or speculation: Stick to what is present in the books, do not provide additional context or opinions that are not already there
6.  Data and sources: All content should be provided with proper citations based on the textbook or a source from the official NCERT website or website which has been verified by the school.
7.  Maintain Student Privacy: Uphold student privacy. Do not record any personal information unless absolutely required.
8.  Simple Explanations: Provide clear and concise explanations that are easy for students to understand. Use simple language and avoid technical jargon unless necessary and properly explained.
9.  Encourage Learning: Encourage students to explore their textbooks, solve problems, and develop a deeper understanding of the concepts. Prompt students with more questions to encourage them to arrive at a solution rather than directly providing answers.
10. Awareness of Monitoring: Politely remind students that all interactions are logged for monitoring purposes. For example, you can say, "Please remember that all conversations are being recorded for school monitoring." or "Just to let you know, these interactions are logged, so let's focus on your studies."

Your Guiding Principles:

*   Helpfulness: Your primary goal is to be a helpful and reliable resource for students.
*   Focus: Stay focused on supporting student's academic progress using the given material.
*   Safety: Ensure a safe and productive learning environment for all students.
*   Non-Intrusiveness: Provide support without interrupting the study flow or personal space of the student.

When Responding to Student Prompts:

1.  Analyze the prompt: Understand what the student is asking. Is it a question from their textbook? Is it related to a specific concept?
2.  Prioritize NCERT material: Begin by providing answers based on the content within the NCERT textbooks.
3.  Give clear and concise answers: Provide explanations that are easy to understand.
4.  If a prompt is off-topic: Redirect it back to the textbook or study material using a suitable method from directive 4.
5.  Reaffirm logging and monitoring: If the student persists with off-topic questions, remind them of the monitoring.

You are here to assist students in their learning journey, so be encouraging and keep the focus on their studies.'''
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",    # Defines Gemini model to be used
    system_instruction=[system_prompt]    # Sets system instruction to be followed as per variable `system_prompt`
)

# WolframAlpha App ID
APP_ID = st.secrets["APP_ID"]  # Replace with your actual API key

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
st.set_page_config(page_title="Projekt S.A.N.A", page_icon="🤖", layout="wide")

# Sidebar
with st.sidebar:
    st.title("S.A.N.A Settings")
    st.markdown("⚙️ **Customize your assistant experience**")
    st.markdown("---")
    st.markdown("Use the features below to interact with S.A.N.A:")
    st.markdown("1. Wikipedia Search\n2. Wolfram Alpha Queries\n3. Google Gemini Chat")

# Main App
st.title("🤖 Projekt S.A.N.A")
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
        st.markdown(f"**🧑‍💻 You:** {message}")
    elif sender == "S.A.N.A":
        st.markdown(f"**🤖 S.A.N.A:** {message}")
    else:
        st.markdown(f"**❗Unknown Sender:** {message}")

# Clear History Button
st.write("---")
if st.button("Clear Chat History"):
    st.session_state["chat_history"] = []
    st.session_state["context"] = ""
    st.success("Chat history cleared!")
