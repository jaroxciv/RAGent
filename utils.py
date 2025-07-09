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
anthropic_client = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
# Load Mistral API Key (Mistral client will be initialized on demand)
mistral_api_key = os.getenv('MISTRAL_API_KEY')

# Determine LLM Provider
# Default to Mistral if not set, due to Claude credit issues mentioned by user.
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'mistral').lower()
logging.info(f"[utils] Using LLM Provider: {LLM_PROVIDER}")

# Placeholder for Mistral client, will be initialized in its specific function
mistral_client = None

# Internal function for Claude RAG
def _call_claude_rag_internal(query, context, dry_run=False):
    if dry_run: # This dry_run check is specific to this function's operation
        logging.info("[_call_claude_rag_internal] Dry run: Returning mock Claude response.")
        return "Mock Claude RAG response: This is a dry run response."

    if not anthropic_client.api_key:
        logging.error("[_call_claude_rag_internal] Anthropic API key not configured.")
        return "Error: Anthropic API key not configured."

    logging.debug(f"[_call_claude_rag_internal] Processing query for Claude: {query[:100]}...")
    system_prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
    If the context doesn't contain enough information to answer confidently, indicate that.
    When referencing information from the podcasts, please cite the source (e.g., "Episode 109", "Sham at [00:00:52.05]") if available in the context.
    Focus on extracting key insights, topics, and speakers mentioned in the podcast excerpts.
    
    Context:
    {context}
    """

    try:
        message = client.messages.create(
            model="claude-3-opus-20240229", # Consider making model configurable too in future
            max_tokens=1024, # Increased max_tokens for potentially more detailed answers
            system=system_prompt,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        logging.debug("[_call_claude_rag_internal] Successfully received response from Claude.")
        # Ensure message.content is a string, handle list of blocks if necessary
        if isinstance(message.content, list) and len(message.content) > 0 and hasattr(message.content[0], 'text'):
            return "".join([block.text for block in message.content if hasattr(block, 'text')])
        return str(message.content) # Fallback, though typically list of TextBlock
    except Exception as e:
        logging.error(f"[_call_claude_rag_internal] Error calling Claude API: {e}")
        return f"Error calling Claude API: {e}"

# Internal function for Mistral RAG
def _call_mistral_rag_internal(query, context, dry_run=False):
    global mistral_client # Use the global client instance
    if dry_run:
        logging.info("[_call_mistral_rag_internal] Dry run: Returning mock Mistral response.")
        return "Mock Mistral RAG response: This is a dry run response."

    if not mistral_api_key:
        logging.error("[_call_mistral_rag_internal] Mistral API key not configured.")
        return "Error: Mistral API key not configured."

    try:
        from mistralai.client import MistralClient
        from mistralai.models.chat_completion import ChatMessage
    except ImportError:
        logging.error("[_call_mistral_rag_internal] Mistral AI client library not installed. Please run `pip install mistralai`.")
        return "Error: Mistral AI client library not installed."

    if mistral_client is None:
        mistral_client = MistralClient(api_key=mistral_api_key)

    logging.debug(f"[_call_mistral_rag_internal] Processing query for Mistral: {query[:100]}...")

    # System prompt construction for Mistral (can be adapted)
    # Mistral models often use a simpler system prompt or integrate it into the user message.
    # For now, we'll keep a similar structure but note it might need tuning.
    # The main instruction about using context and citing sources is important.
    system_message_content = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so.
When referencing information from the podcasts, please cite the source (e.g., "Episode 109", "Sham at [00:00:52.05]") if available in the context.
Focus on extracting key insights, topics, and speakers mentioned in the podcast excerpts.
Provided context:
{context}"""

    user_message_content = f"Question: {query}"

    messages = [
        # Some Mistral models prefer system prompt as the first user message or have specific ways to handle it.
        # A common pattern is User/Assistant turns. We can provide the system instructions as part of the first user message.
        # Or use the system role if the model supports it well (e.g. mistral-large).
        # For simplicity and broad compatibility, let's combine system instructions with the user query for now.
        # A more advanced setup might use model-specific prompt formatting.
        ChatMessage(role="user", content=f"{system_message_content}\n\n{user_message_content}")
    ]

    try:
        # Using a common Mistral model, e.g., 'mistral-small-latest' or 'mistral-large-latest'
        # 'open-mistral-7b' is also an option if you use their open models via appropriate endpoints.
        # For platform.mistral.ai, models like 'mistral-small-latest', 'mistral-medium-latest', 'mistral-large-latest'.
        # Let's use a generally available one, assuming platform API.
        chat_response = mistral_client.chat(
            model="mistral-small-latest", # This can be made configurable
            messages=messages,
            # max_tokens can be set if needed, similar to Claude.
        )
        logging.debug("[_call_mistral_rag_internal] Successfully received response from Mistral.")
        if chat_response.choices and len(chat_response.choices) > 0:
            return chat_response.choices[0].message.content
        else:
            logging.warning("[_call_mistral_rag_internal] Mistral response was empty or malformed.")
            return "Error: Received an empty or malformed response from Mistral."
    except Exception as e:
        logging.error(f"[_call_mistral_rag_internal] Error calling Mistral API: {e}")
        return f"Error calling Mistral API: {e}"


# Generic LLM RAG dispatcher function
def call_llm_rag(query, context, dry_run=False):
    if dry_run: # This top-level dry_run check can return a generic mock before provider selection
        logging.info(f"[call_llm_rag] Dry run mode for provider {LLM_PROVIDER} - returning generic mock response.")
        return f"Mock RAG response for {LLM_PROVIDER}: This is a dry run."

    logging.info(f"[call_llm_rag] Routing to LLM Provider: {LLM_PROVIDER}")
    if LLM_PROVIDER == 'claude':
        return _call_claude_rag_internal(query, context, dry_run=False) # dry_run handled by top-level
    elif LLM_PROVIDER == 'mistral':
        return _call_mistral_rag_internal(query, context, dry_run=False) # dry_run handled by top-level
    else:
        logging.error(f"[call_llm_rag] Unknown LLM_PROVIDER: {LLM_PROVIDER}")
        return "Error: Unknown LLM provider configured."

# Removed call_tavily_web_search function
# Removed assess_confidence function
# Removed synthesize_information function