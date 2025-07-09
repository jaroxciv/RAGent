import streamlit as st
from app import main
import os

st.set_page_config(page_title="The King RAGent", page_icon="👑", layout="wide")

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main_chat():
    # Create two columns: sidebar and main content
    with st.sidebar:
        st.title("Settings")
        dry_run = st.checkbox("🔧 Dry Run Mode", value=False, help="Run without making API calls")
        if dry_run:
            st.warning("Test mode active - using stub responses")
    
    # Main content
    st.title("👑 The King RAGent")
    st.markdown("### Your AI Research Assistant")
    
    initialize_session_state()
    
    # Directory Input for PDFs
    if not st.session_state.initialized:
        st.subheader("Initialize Podcast Library")
        pdf_directory = st.text_input("Enter the path to the directory containing podcast PDFs:", key="pdf_dir_input")

        if pdf_directory:
            if os.path.isdir(pdf_directory):
                with st.spinner(f"Processing PDFs from {pdf_directory}..."):
                    response = main(None, initialize=True, directory_path=pdf_directory, dry_run=dry_run)
                st.success(response) # Display success or error message from main
                if "Error" not in response:
                    st.session_state.initialized = True
                    st.rerun() # Rerun to hide the input and show chat
                else:
                    st.error(response)
            elif pdf_directory != "": # if user entered something but it's not a valid directory
                st.error("Invalid directory path. Please enter a valid path.")
        st.markdown("---") # Separator

    # Display chat history only if initialized
    if st.session_state.initialized:
        display_chat_history()
    
        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = main(prompt, initialize=False, dry_run=dry_run)
                
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the chat display
        st.rerun()

if __name__ == "__main__":
    main_chat()