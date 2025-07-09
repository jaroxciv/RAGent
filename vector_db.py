# Set environment variable before imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
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
                except Exception as e:
            # If there's an issue loading, it might be due to version conflicts or corruption.
            # It's safer to log the error and proceed as if no DB exists.
            logging.warning(f"[__init__] Could not load existing vector store: {e}. Will re-initialize if data is provided.")
                    self.db = None
            else:
                self.db = None
        else:
            self.db = None
            logging.info("[__init__] Dry run mode - skipping DB load")

    def initialize_from_directory(self, directory_path, chunk_size=300, chunk_overlap=50):
        """Initialize vector DB with PDF content from a directory."""
        if self.dry_run:
            logging.info(f"[initialize_from_directory] Dry run mode - skipping PDF initialization for {directory_path}")
            return True

        logging.debug(f"[initialize_from_directory] Loading PDFs from directory: {directory_path}")
        all_docs = []
        current_processed_files = []
        try:
            for filename in os.listdir(directory_path):
                if filename.lower().endswith(".pdf"):
                    pdf_path = os.path.join(directory_path, filename)
                    logging.debug(f"[initialize_from_directory] Loading PDF: {pdf_path}")
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()
                    logging.debug(f"[initialize_from_directory] Loaded {len(documents)} pages from {filename}")
                    current_processed_files.append(filename)

                    # Add metadata to each document
                    for doc in documents:
                        doc.metadata["source"] = filename # Store filename as source
                        # You could add more metadata here, e.g., episode number if parsable from filename

                    text_splitter = CharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separator="\n"
                    )
                    split_docs = text_splitter.split_documents(documents)
                    all_docs.extend(split_docs)
                    logging.debug(f"[initialize_from_directory] Split {filename} into {len(split_docs)} chunks")

            if not all_docs:
                logging.warning(f"[initialize_from_directory] No PDF files found in directory: {directory_path}")
                # Even if no new files are processed, retain existing DB if any.
                # Return True if db exists, False otherwise, or based on specific logic.
                return bool(self.db)

            # If re-initializing, it's often best to clear old entries or use a new collection.
            # For simplicity here, we'll overwrite by creating a new Chroma instance.
            # This means old data is gone if new PDFs are added.
            # A more sophisticated approach would be to add_documents and handle updates/deletions.

            self.db = Chroma.from_documents(
                documents=all_docs,
                embedding=self.embedding_function,
                persist_directory=self.persist_directory,
                client_settings=chromadb.config.Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            self.processed_files = current_processed_files # Update the list of processed files
            logging.info(f"[initialize_from_directory] Vector store initialized successfully with {len(all_docs)} chunks from {len(current_processed_files)} files in {directory_path}")
            return True

        except Exception as e:
            logging.error(f"[initialize_from_directory] Error initializing vector store: {e}")
            return False

    def get_processed_files(self):
        """Return the list of processed PDF filenames."""
        return self.processed_files

    def search(self, query, n_results=3):
        """Search the vector store for relevant documents."""
        if self.dry_run:
            logging.info(f"[search] Dry run mode - returning mock results for query: {query}")
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