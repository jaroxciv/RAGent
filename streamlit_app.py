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
    st.session_state.dry_run_mode = dry_run # Store dry_run_mode in session state from sidebar

    # Automatic Initialization from preset directory
    if not st.session_state.get("initialized", False): # Use .get for safety
        st.subheader("Podcast Library Status")
        # Access PDF_DIRECTORY from app.py. A bit indirect, but avoids re-defining.
        # For cleaner access, PDF_DIRECTORY could be in a shared config.
        pdf_dir_path_display = main.__globals__.get('PDF_DIRECTORY', './podcast_pdfs/') # Default if not found

        with st.spinner(f"Checking and initializing from preset PDF directory ({pdf_dir_path_display})..."):
            # Call main with initialize=True. main will use its PDF_DIRECTORY.
            # Pass the current dry_run state from the session.
            init_response = main(user_query=None, initialize=True, dry_run=st.session_state.dry_run_mode)

        # Check specific phrases that indicate successful processing and readiness
        if "Successfully processed" in init_response and "Ready for questions." in init_response:
            st.session_state.initialized = True
            st.success(init_response)
            import time
            time.sleep(1) # Short delay for user to see message
            st.rerun()
        elif "PDF directory" in init_response and "was created" in init_response:
            st.info(init_response)
        elif "No PDF files found" in init_response:
            st.warning(init_response)
        elif "Error" in init_response: # Generic error catch
            st.error(init_response)
        else:
            st.info(init_response) # Default message display

    # Display chat interface only if initialized
    if st.session_state.get("initialized", False):
        display_chat_history()

        if prompt := st.chat_input("What would you like to know about the podcasts?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Pass current dry_run state for query processing
                    response = main(prompt, initialize=False, dry_run=st.session_state.dry_run_mode)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            # st.rerun() # Usually not needed for chat, but uncomment if updates are inconsistent
    else:
        # This block will be shown if initialization hasn't successfully completed.
        st.markdown("---")
        st.info(f"Podcast library not yet ready. Please ensure PDF files are in the `podcast_pdfs` directory. Refresh the page to re-attempt initialization.")

if __name__ == "__main__":
    main_chat()