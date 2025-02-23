import streamlit as st
import random
import chatbot  # Import the chatbot backend

# Streamlit Page Config
st.set_page_config(page_title="VBot Chat", page_icon="🤖", layout="centered")

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": "Hello! Ask me anything about VIT."}
    ]

st.title("🤖 VBot Chat")

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Chat Display using Streamlit's built-in chat UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
user_input = st.chat_input("Type your message...")

# Handle user input
if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Keyword Matching for Greetings
    GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
    GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

    if user_input != 'bye':
        if user_input in ('thanks', 'thank you'):
            bot_reply = " You are welcome.."
        else:
            if greeting(user_input):
                bot_reply = greeting(user_input)
            else:
                bot_reply = chatbot.response(user_input)
    else:
        bot_reply = "ROBO: Bye! Take care.."

    # Append bot reply
    st.session_state.messages.append({"role": "bot", "content": bot_reply})

    # Refresh UI
    st.rerun()

