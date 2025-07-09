import time
from vector_db import VectorStore
from utils import call_claude_rag # Removed assess_confidence, synthesize_information, call_tavily_web_search
import logging

def main(user_query, initialize=False, directory_path=None, dry_run=False):
    logging.debug(f"[main] Starting process with query: {user_query[:100]}...")
    if dry_run:
        logging.info("[main] Running in dry run mode")
    
    vector_store = VectorStore(persist_directory="./chroma_db", dry_run=dry_run)

    if initialize and directory_path:
        logging.info(f"[main] Initializing with PDFs from directory: {directory_path}")
        if vector_store.initialize_from_directory(directory_path):
            return "PDFs processed and ready for questions!"
        else:
            return "Error processing PDFs. Check logs for details."
    elif not vector_store.db and not initialize: # Check if DB exists or if we are initializing
        # This condition means we are not initializing and the DB isn't loaded.
        # This could be because it's the first run, or loading failed, or no directory was processed yet.
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
    
    logging.debug("[main] Getting RAG response")
    rag_response = call_claude_rag(user_query, internal_response, dry_run)
    
    if isinstance(rag_response, list): # Ensure response is a string
        rag_response = " ".join(str(item) for item in rag_response)
    
    # Directly return the RAG response. Web search and synthesis are removed.
    logging.info("[main] Returning RAG response")
    return rag_response