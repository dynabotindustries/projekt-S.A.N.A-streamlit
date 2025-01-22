import streamlit as st
import wikipedia
import wolframalpha
import google.generativeai as genai

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
st.set_page_config(page_title="Projekt S.A.N.A for RMK School", page_icon=logo, layout="wide")

# Sidebar
with st.sidebar:
    st.title("S.A.N.A Settings")
    st.markdown("⚙️ **Customize your assistant experience (coming soon!)**")
    st.markdown("---")
    st.markdown("Use the features below to interact with S.A.N.A:")
    st.markdown("1. Wikipedia Search\n2. Wolfram Alpha Queries\n3. Google Gemini Chat")

# Main App

# Logo and Title in HTML format for inline logo
st.markdown(f"<h1><img src='{logo}' width=70 style='display:inline-block; margin-right:15px'></img><b>Projekt S.A.N.A fir RMK School:</b></h1>", unsafe_allow_html=True)

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
