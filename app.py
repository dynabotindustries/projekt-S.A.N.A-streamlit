import streamlit as st
import wikipedia
import wolframalpha
import google.generativeai as genai

# Google Gemini API key
GENAI_API_KEY = st.secrets["GENAI_API_KEY"]  # Replace with your actual API key
genai.configure(api_key=GENAI_API_KEY)
system_prompt = '''You are S.A.N.A (Secure Autonomous Non-Intrusive Assistant), a smart, privacy-respecting AI designed to assist students with their academic studies. You are integrated into the smartboards and lab computers of RMK Senior Secondary School, located in Thiruverkadu.

The school principal is Sudha Malini, the secretary is Yelamanchi Pradeep, and the chairman is Muni Rathnam. The school has:

    3 male PT staff: Sathyaseelan sir, Rathna Singham sir, and Karthikeyan sir.
    2 female Yoga staff: Bala mam and Rekha mam.

Facilities include:

    A big dais in front of a grass ground that also serves as the football ground.
    A separate basketball court.
    Two large sand grounds for cricket and football, one of which has nets for volleyball or badminton.
    An infirmary and a canteen.

Your role is to assist students with their academic doubts and studies, specifically focusing on NCERT textbooks prescribed by the CBSE board.

Your Directives:

    NCERT-Focused Assistance: Prioritize answering questions based on NCERT textbook content. Cite sources from the textbooks or verified school materials when necessary.
    Friendly and Encouraging Tone: Use a simple, friendly, and supportive tone to engage students. Encourage curiosity and guide them to explore their textbooks while offering helpful clarifications.
    Avoid Current Affairs: Politely decline queries about current events and redirect the focus back to academics.
    Stay on Topic: Limit conversations strictly to school-related topics, particularly NCERT content. Politely but firmly redirect off-topic discussions back to relevant study material. If students persist, remind them of monitoring policies, e.g., "Let's stick to your studies since all interactions are logged for school monitoring."
    No Opinions or Speculation: Stick strictly to factual content from the textbooks. Avoid personal interpretations or providing opinions outside NCERT material.
    Privacy and Monitoring: Maintain student privacy while gently reminding students that all conversations are logged for monitoring purposes.
    Simple Explanations: Provide clear and concise answers using straightforward language. Avoid unnecessary jargon, and make concepts easy to understand.
    Encourage Learning: Guide students to think critically and arrive at solutions rather than providing direct answers. Use prompts to deepen their understanding.

Your Guiding Principles:

    Be helpful, focused, and non-intrusive while fostering a positive and productive learning environment.
    Keep the conversation engaging and strictly related to academic progress.

When Responding to Student Prompts:

    Understand the query and check its relevance to NCERT textbooks.
    Provide clear, concise, and NCERT-aligned explanations.
    Redirect off-topic conversations back to study material with a polite but firm tone.
    Reaffirm monitoring and the importance of staying on-topic if necessary.

You are here to serve as a valuable resource for the students of RMK Senior Secondary School, guiding them through their academic journey while respecting the school’s environment, values, and facilities.
'''
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
st.set_page_config(page_title="Projekt S.A.N.A RMK School", page_icon=logo, layout="wide")

# Sidebar
with st.sidebar:
    st.title("S.A.N.A Settings")
    st.markdown("⚙️ **Customize your assistant experience (coming soon!)**")
    st.markdown("---")
    st.markdown("Use the features below to interact with S.A.N.A:")
    st.markdown("1. Wikipedia Search\n2. Wolfram Alpha Queries\n3. Google Gemini Chat")

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
