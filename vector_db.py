# Set environment variable before imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Updated import
import chromadb

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2", persist_directory="./chroma_db", dry_run=False):
        """Initialize the vector store with the specified embedding model."""
        self.dry_run = dry_run
        logging.debug(f"[__init__] Initializing VectorStore with model: {model_name}, dry_run: {self.dry_run}")
        self.embedding_function = HuggingFaceEmbeddings(model_name=model_name)
        self.persist_directory = persist_directory
        self.processed_files = [] # To store names of processed files
        
        if not self.dry_run:
            # Try to load existing DB
            if os.path.exists(persist_directory):
                try:
                    self.db = Chroma(
                        persist_directory=persist_directory,
                        embedding_function=self.embedding_function,
                        client_settings=chromadb.config.Settings(
                            anonymized_telemetry=False,
                            is_persistent=True
                        )
                    )
                    logging.info("[__init__] Loaded existing vector store from disk")
                    # Attempt to load processed files list if DB exists
                    # This is a simple way; a more robust way would be to store this list in a separate file or DB metadata
                    if self.db:
                        # Get all unique 'source' metadata values
                        all_entries = self.db.get(include=["metadatas"])
                        if all_entries and all_entries['metadatas']:
                            unique_sources = set(meta['source'] for meta in all_entries['metadatas'] if 'source' in meta)
                            self.processed_files = list(unique_sources)
                    logging.info(f"[__init__] Loaded {len(self.processed_files)} processed file names from existing DB.")
                except Exception as e: # pylint: disable=broad-except
                    # If there's an issue loading, it might be due to version conflicts or corruption.
                    # It's safer to log the error and proceed as if no DB exists.
                    logging.warning(f"[__init__] Could not load existing vector store: {e}. Will re-initialize if data is provided.")
                    self.db = None
            else:
                self.db = None
        else:
            self.db = None
            logging.info("[__init__] Dry run mode - skipping DB load")

    def initialize_from_directory(self, directory_path, chunk_size=1000, chunk_overlap=200): # Updated defaults
        """Initialize vector DB with PDF content from a directory."""
        if self.dry_run:
            logging.info(f"[initialize_from_directory] Dry run mode - skipping PDF initialization for {directory_path} with chunk_size={chunk_size}")
            return True

        logging.debug(f"[initialize_from_directory] Initializing from directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            logging.warning(f"[initialize_from_directory] Directory not found: {directory_path}. Cannot initialize.")
            # If DB already loaded from persistence, it's still valid.
            # self.processed_files would be from persisted DB.
            return bool(self.db) 

        all_docs_to_split = [] # Will collect all page documents with prepended source info
        current_processed_files = []
        
        # Regex to find Episode number and YouTube link (simple version)
        import re
        episode_regex = re.compile(r"Episode\s*#?\s*(\d+)", re.IGNORECASE)
        youtube_regex = re.compile(r"(https?://youtu\.be/[^\s]+)", re.IGNORECASE)

        try:
            for filename in os.listdir(directory_path):
                if filename.lower().endswith(".pdf"):
                    pdf_path = os.path.join(directory_path, filename)
                    current_processed_files.append(filename)
                    loader = PyPDFLoader(pdf_path)
                    pages = loader.load() # List of Document objects (one per page)
                    
                    if not pages:
                        logging.warning(f"[initialize_from_directory] No content loaded from {filename}")
                        continue

                    # Attempt to parse Episode & YouTube link from the first page
                    first_page_text = pages[0].page_content
                    episode_match = episode_regex.search(first_page_text[:500]) # Search in first 500 chars
                    youtube_match = youtube_regex.search(first_page_text[:500])
                    
                    episode_info = f"Episode {episode_match.group(1)}" if episode_match else "Unknown Episode"
                    youtube_link_info = youtube_match.group(1) if youtube_match else "Unknown YouTube Link"
                    source_prefix = f"Source Information: {episode_info} - {youtube_link_info}\n\n"
                    
                    # Prepend source info to each page's content from this PDF
                    for page_doc in pages:
                        page_doc.page_content = source_prefix + page_doc.page_content
                        page_doc.metadata["source_filename"] = filename # Keep original filename in metadata
                        # Add parsed info to metadata too, if useful for direct filtering later (optional)
                        page_doc.metadata["episode_number"] = episode_match.group(1) if episode_match else "Unknown"
                        page_doc.metadata["youtube_link"] = youtube_match.group(1) if youtube_match else "Unknown"
                        all_docs_to_split.append(page_doc)

            if not all_docs_to_split:
                logging.warning(f"[initialize_from_directory] No processable PDF content found in {directory_path}.")
                # If no files in directory, but DB was loaded from persistence, consider it "successful"
                # as the existing DB is preserved. self.processed_files remains from __init__.
                if self.db:
                    logging.info("[initialize_from_directory] No new PDFs found, existing DB preserved.")
                    return True 
                # If no DB and no files, it's an empty state.
                self.processed_files = [] # Ensure processed_files is empty
                return False # Indicate nothing was initialized and no prior DB.

            # Now, split the processed documents (pages with prepended source info) into chunks
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n" # Default separator, might need adjustment based on transcript format
            )
            final_chunks = text_splitter.split_documents(all_docs_to_split)
            logging.debug(f"[initialize_from_directory] Split {len(all_docs_to_split)} pages into {len(final_chunks)} final chunks.")

            if not final_chunks:
                logging.warning(f"[initialize_from_directory] Text splitting resulted in no final chunks for {directory_path}.")
                # Preserve existing DB if it exists, similar to no PDFs found
                return bool(self.db)

            # If new documents are found and chunked, (re)create the DB with them.
            logging.info(
                f"[initialize_from_directory] Processed {len(current_processed_files)} PDF(s), "
                f"resulting in {len(final_chunks)} chunks. Rebuilding vector store.")
            self.db = Chroma.from_documents(
                documents=final_chunks, # Use the final chunks with prepended source info
                embedding=self.embedding_function,
                persist_directory=self.persist_directory,
                client_settings=chromadb.config.Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            self.processed_files = current_processed_files # Update the list of processed files
            logging.info(
                f"[initialize_from_directory] Vector store initialized successfully with {len(all_docs_to_split)} "
                f"chunks from {len(current_processed_files)} files in {directory_path}"
                )
            return True

        except Exception as e:
            logging.error(f"[initialize_from_directory] Error initializing vector store: {e}")
            return False

    def get_processed_files(self):
        """Return the list of processed PDF filenames."""
        return self.processed_files

    def search(self, query, n_results=5): # Updated default
        """Search the vector store for relevant documents."""
        if self.dry_run:
            logging.info(f"[search] Dry run mode - returning mock results for query: {query} with n_results={n_results}")
            return "Mock result: This is a dry run response."

        logging.debug(f"[search] Searching for: {query}")
        try:
            if not self.db:
                logging.error("[search] Vector store not initialized")
                return ""

            docs = self.db.similarity_search(query, k=n_results)
            context = "\n".join(doc.page_content for doc in docs)
            logging.debug(f"[search] Found {len(docs)} relevant documents")
            return context

        except Exception as e:
            logging.error(f"[search] Error searching vector store: {e}")
            return ""

    def __del__(self):
        """Cleanup when the object is destroyed."""
        logging.debug("[__del__] Vector store cleanup complete")