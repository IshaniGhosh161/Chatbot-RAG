import streamlit as st
# from auth import Authentication
# from chat import ChatBot
from main import Authentication, ChatBot

def main():
    # Set page configuration
    st.set_page_config(page_title="Chatbot App", layout="wide")

    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['show_registration'] = False

    # Authentication handler
    auth = Authentication()

    # Main app logic
    if not st.session_state['logged_in']:
        # Render login or registration page
        auth.render()
    else:
        # Sidebar logout
        if st.sidebar.button("Logout"):
            auth.logout()
            st.rerun()

        # Instantiate ChatBot
        # chatbot = ChatBot(st.session_state['username'])
        chatbot = ChatBot()

        # Create sidebar with chat sessions
        chatbot.sidebar_chat_sessions()

        # Main chat interface
        st.title(f"Welcome, {st.session_state['username']}!")
        chatbot.chat_interface()

if __name__ == "__main__":
    main()