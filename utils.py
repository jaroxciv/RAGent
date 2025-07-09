import time
from anthropic import Anthropic
import logging
import requests

# from tavily import TavilyClient # Removed
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the Anthropic client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def call_claude_rag(query, context, dry_run=False):
    if dry_run:
        logging.info("[call_claude_rag] Dry run mode - returning mock response")
        return "Mock response: This is a dry run response."

    logging.debug(
        f"[call_claude_rag] Processing query: {query[:100]}..."
    )  # Log first 100 chars of query
    system_prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
    If the context doesn't contain enough information to answer confidently, indicate that.
    When referencing information from the podcasts, please cite the source (e.g., "Episode 109", "Sham at [00:00:52.05]") if available in the context.
    Focus on extracting key insights, topics, and speakers mentioned in the podcast excerpts.
    
    Context:
    {context}
    """

    try:
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": query}],
        )
        logging.debug("[call_claude_rag] Successfully received response from Claude")
        return message.content
    except Exception as e:
        logging.error(f"[call_claude_rag] Error calling Claude API: {e}")
        return ""


# Removed call_tavily_web_search function
# Removed assess_confidence function
# Removed synthesize_information function
