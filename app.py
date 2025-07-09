import time
from vector_db import VectorStore
from utils import call_llm_rag # Changed from call_claude_rag
import logging
import os

# Define the preset directory for podcast PDFs
PDF_DIRECTORY = "./podcast_pdfs/"

def main(user_query, initialize=False, directory_path=None, dry_run=False): # directory_path can be kept for flexibility or future use
    if user_query:
        logging.debug(f"[main] Starting process with query: {user_query[:100]}...")
    else:
        logging.debug("[main] Starting process (no user query, likely initialization or internal call)...")

    # Use preset directory if initialize is true and no specific directory_path is given
    # Or if we are just running a query and need to ensure VectorStore loads from the correct default place
    # For Streamlit auto-initialization, directory_path parameter will be None.

    if dry_run:
        logging.info("[main] Running in dry run mode")

    # VectorStore will persist to "./chroma_db". PDF_DIRECTORY is where it reads from.
    vector_store = VectorStore(persist_directory="./chroma_db", dry_run=dry_run)

    if initialize:
        # Always use the preset PDF_DIRECTORY for initialization calls.
        logging.info(f"[main] Attempting initialization/update from preset directory: {PDF_DIRECTORY}")
        if not os.path.exists(PDF_DIRECTORY):
            os.makedirs(PDF_DIRECTORY, exist_ok=True) # Create directory if it doesn't exist
            logging.info(f"[main] Created PDF directory: {PDF_DIRECTORY}")
            return f"PDF directory '{PDF_DIRECTORY}' was created. Please add your podcast PDFs there and refresh."

        if vector_store.initialize_from_directory(PDF_DIRECTORY):
            processed_files = vector_store.get_processed_files()
            if processed_files:
                return f"Successfully processed {len(processed_files)} PDF(s) from '{PDF_DIRECTORY}'. Ready for questions."
            else:
                return f"No PDF files found or processed in '{PDF_DIRECTORY}'. Please add PDFs and refresh."
        else:
            return f"Error processing PDFs from '{PDF_DIRECTORY}'. Check logs for details."

    # This block handles the case where we are not initializing, but querying.
    # It checks if the vector store (db) is loaded.
    # If db is not loaded, it could be the first run or a failed load.
    if not vector_store.db and not initialize:
        logging.error("[main] Vector store not initialized or no PDFs processed.")
        # Try to load processed files list to see if it was initialized before but just not loaded in this session
        processed_files = vector_store.get_processed_files()
        if processed_files:
             return f"Vector store is initialized with {len(processed_files)} file(s). Ready for questions. Loaded files: {', '.join(processed_files)}"
        return "Error: Vector store not initialized. Please provide a directory of PDF files first."
    elif initialize: # This means initialize is True but directory_path was not provided
        logging.error("[main] Directory path not provided for initialization")
        return "Please provide a directory path to initialize the system."

    # Handle special commands
    if user_query.strip().lower() in ["list podcasts", "list all podcasts", "what podcasts are available?", "list available podcasts"]:
        processed_files = vector_store.get_processed_files()
        if not processed_files:
            return "No podcasts have been processed yet. Please initialize with a directory of PDFs."
        response = "Available podcasts:\n" + "\n".join([f"- {f}" for f in processed_files])
        # Optionally, could add a summary here using an LLM call if desired for "what topics are covered?"
        return response

    if not vector_store.db: # Final check if DB is not available for querying
        logging.error("[main] Vector store is not queryable right now.")
        return "Error: Vector store is not ready for querying. Please initialize or check logs."

    logging.debug("[main] Starting query processing for: " + user_query)
    internal_response = vector_store.search(user_query)

    logging.debug("[main] Getting RAG response from configured LLM provider.")
    rag_response = call_llm_rag(user_query, internal_response, dry_run) # Changed here

    # The RAG functions should ideally return a string directly.
    # If they might return lists (e.g. Claude's TextBlock list), ensure it's handled.
    # _call_claude_rag_internal was updated to handle this. _call_mistral_rag_internal returns string.
    if not isinstance(rag_response, str):
        logging.warning(f"[main] RAG response was not a string, attempting conversion. Type: {type(rag_response)}")
        rag_response = str(rag_response) # General conversion if not already a string

    # Directly return the RAG response. Web search and synthesis are removed.
    logging.info("[main] Returning RAG response")
    return rag_response