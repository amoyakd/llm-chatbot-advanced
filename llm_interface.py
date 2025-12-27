import os
import time
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()  # Loads .env file automatically

class HuggingFaceInferenceAPI:
    """
    Manages interactions with Hugging Face Inference API using the official InferenceClient.
    """
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get("HF_TOKEN")
        if not self.api_token:
            raise ValueError(
                "HF_TOKEN not found. Please set the HF_TOKEN environment variable or pass it as an argument."
            )
        # Using the newer provider system for Llama 3.2
        self.client = InferenceClient(
            provider="auto", 
            api_key=self.api_token
        )
        self.model = "meta-llama/Llama-3.2-3B-Instruct"
        # self.max_retries = 3
        # self.base_wait_time = 1  # seconds

    def _generate_text(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Generate text using the InferenceClient with retry logic.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated text.
        """
        try:
            # Use the OpenAI-compatible syntax for better reliability
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                stream=False
            )
            # Correctly extract the string content from the response object
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return f"Error generating response: {e}"

    def moderate_query(self, query: str) -> bool:
        # """
        # Moderates a query using Llama-Guard-3-1B via Hugging Face Inference API.

        # Args:
        #     query: The user's query.

        # Returns:
        #     True if the query is safe, False otherwise.
        # """
        # # try:
        # #     prompt = f"[INST] Is this prompt safe and appropriate for a product inquiry chatbot? '{query}' Answer only with 'safe' or 'unsafe'. [/INST]"
        # #     response = self._generate_text(prompt, max_tokens=10)
            
        # #     if "unsafe" in response.lower():
        # #         print(f"Query flagged as unsafe: {query}")
        # #         return False
        # #     return True
        # # except Exception as e:
        # #     print(f"Error during moderation: {e}")
        # #     # Default to safe to avoid blocking users if the moderation model fails.
        # #     return True
        
        # # Llama Guard 3 expects this specific instruction format to be accurate
        # moderation_prompt = f"""[INST] Task: Check if there is unsafe content in 'User' messages \
        #     in accordance with our safety policy.
        
        # <BEGIN UNSAFE CONTENT CATEGORIES>
        # S1: Violent Crimes. 
        # S2: Non-Violent Crimes. 
        # S3: Sexually Explicit Content.
        # S4: Child Sexual Exploitation.
        # S5: Defamation.
        # S6: Specialized Advice.
        # S7: Public Interest Topics.
        # S8: Harassment.
        # <END UNSAFE CONTENT CATEGORIES>

        # User: {query} [/INST]"""

        # try:
        #     print("Sending moderation request...")
        #     # We call the specialized guard model instead of the general Llama 3.2 model here
        #     response = self.client.chat.completions.create(
        #         model="meta-llama/Llama-Guard-3-1B",
        #         messages=[{"role": "user", "content": moderation_prompt}],
        #         max_tokens=10
        #     )
            
        #     # Llama-Guard-3 returns "safe" or "unsafe" as its primary output
        #     result = response.choices[0].message.content.strip().lower()
        #     print(f"Moderation result for query '{query}': {result}")
            
        #     # If the word 'unsafe' appears, we flag it. Otherwise, it's safe.
        #     return "unsafe" not in result
            
        # except Exception as e:
        #     print(f"Moderation API Error: {repr(e)}")
        #     # Default to True (safe) so the user isn't blocked by a minor API hiccup
        #     return True
        
        
        """
        Moderates a query using a stable, high-availability model (Qwen 2.5).
        """
        # Qwen 2.5 is currently the most reliable for free-tier serverless inference
        moderator_model = "Qwen/Qwen2.5-7B-Instruct"
        
        moderation_prompt = f"""<|im_start|>system
        You are a content moderator. Your job is to classify if a user query is SAFE or UNSAFE.
        - SAFE: General questions, product inquiries, electronics, store help, or friendly chat.
        - UNSAFE: Hate speech, violence, illegal acts, or sexual content.
        Respond with ONLY the word 'SAFE' or 'UNSAFE'.<|im_end|>
        <|im_start|>user
        {query}<|im_end|>
        <|im_start|>assistant"""

        try:
            print(f"Sending moderation request to {moderator_model}...")
            response = self.client.chat.completions.create(
                model=moderator_model,
                messages=[{"role": "user", "content": moderation_prompt}],
                max_tokens=5,
            )
            
            result = response.choices[0].message.content.strip().upper()
            print(f"Moderation result: {result}")
            
            return "UNSAFE" not in result

        except Exception as e:
            # Improved error logging to see exactly what's happening
            print(f"Moderation API Error: {repr(e)}")
            # If the API fails, we assume safe to keep the UX smooth
            return True
    
    
    
    def generate_response(self, query: str, system_prompt: str) -> str:
        """
        Generates a response using Mistral-7B-Instruct via Hugging Face Inference API.

        Args:
            query: The user's query.
            system_prompt: The system prompt with context and instructions.

        Returns:
            The generated response.
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
            
            # Format messages for the model
            formatted_messages = "\n".join(
                [f"<s>[INST] {m['content']} [/INST]" if m["role"] == "user" 
                 else f"{m['content']}" for m in messages]
            )
            
            response = self._generate_text(formatted_messages, max_tokens=500)
            return response.strip()
        except Exception as e:
            print(f"Error during response generation: {e}")
            return "I'm sorry, but I encountered an error while trying to generate a response."

    
    def rewrite_query(self, query: str, system_prompt: str) -> str:
        """
        Rewrites a query using Mistral-7B-Instruct via Hugging Face Inference API.

        Args:
            query: The user's query.
            system_prompt: The system prompt with instructions.

        Returns:
            The rewritten query.
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User query: '{query}'"},
            ]
            
            # Format messages for the model
            formatted_messages = "\n".join(
                [f"<s>[INST] {m['content']} [/INST]" if m["role"] == "user" 
                 else f"{m['content']}" for m in messages]
            )
            
            response = self._generate_text(formatted_messages, max_tokens=200)
            rewritten = response.strip()
            
            # Remove potential quotes around the rewritten query
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            if rewritten.startswith("'") and rewritten.endswith("'"):
                rewritten = rewritten[1:-1]
                
            return rewritten
        except Exception as e:
            print(f"Error during query rewrite: {e}")
            return query  # Fallback to original query on error


# Initialize the API client
_api_client = None

def get_api_client() -> HuggingFaceInferenceAPI:
    """Get or initialize the Hugging Face Inference API client."""
    global _api_client
    if _api_client is None:
        _api_client = HuggingFaceInferenceAPI()
    return _api_client


def moderate_query(query: str) -> bool:
    """
    Moderates a query using Qwen via Hugging Face Inference API.

    Args:
        query: The user's query.

    Returns:
        True if the query is safe, False otherwise.
    """
    print("Moderating query...")
    client = get_api_client()
    return client.moderate_query(query)

def generate_response(query: str, retrieved_docs: list, history: list) -> str:
    """
    Generates a response using Llama-3.2-3B-Instruct via Hugging Face Inference API,
    ensuring it adheres to the retrieved documents.

    Args:
        query: The user's query.
        retrieved_docs: A list of document contents.
        history: The chat history from Gradio.

    Returns:
        The generated response.
    """
    system_prompt = """You are a specialized product inquiry assistant. \
        Your primary and ONLY role is to answer user questions based on \
            the 'Retrieved Documents' provided below.

            Follow these rules strictly:
            1.  Base your entire response on the information found within the 'Retrieved Documents'. \
                Do not use any external knowledge.
            2.  If there are no documents or \
                the documents do not contain the information needed to answer the query, \
                you MUST respond with: \"I'm sorry, but I cannot answer your question with the information I have.\"
            3.  If the documents contain relavant information, use it to construct a clear and concise answer.
                The documents may include metadata such as price, product name, brand, and category.
                The documents may also include product descriptions and features.
                The documents may include customer reviews which can be used to answer questions \
                    about product quality and user satisfaction.
            4.  Some documents may not be fully relevant; \
                carefully select and synthesize information only from the relevant parts.
            5.  Do not fabricate or assume any information not present in the documents.
            6.  Analyze the chat history provided under 'Chat History' for conversational context, \
                but do not use it as a source for answers.
            7.  Respond in a friendly and helpful tone, with concise answers and directly related to the query.\
            8.  Make sure to ask the user relevant follow-up questions.\
            9.  Always format prices with a dollar sign and two decimal places.\
            10. Do not use the term 'Retrieved Documents' in your response. It is only for your reference.
            

            Retrieved Documents:
            ```
            {context}
            ```

            Chat History:
            {chat_history}
            """

    context = "\n\n---\n\n".join(doc for doc in retrieved_docs)
    
    # Format chat history for the prompt
    #formatted_history = "\n".join([f"User: {user_msg}\nAssistant: {bot_msg}" for user_msg, bot_msg in history])
    formatted_history = ""
    for msg in history:
        if msg["role"] == "user":
            formatted_history += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            formatted_history += f"Assistant: {msg['content']}\n"

    prompt = system_prompt.format(context=context, chat_history=formatted_history)

    client = get_api_client()
    return client.generate_response(query, prompt)



def rewrite_query(query: str, history: list) -> str:
    """
    Rewrites a conversational query into a self-contained query using the chat history
    via Hugging Face Inference API.

    Args:
        query: The user's potentially vague query.
        history: The chat history from Gradio.

    Returns:
        A self-contained query.
    """
    system_prompt = """You are an expert at query rewriting. Your task is to rewrite a given 'user query' \
        into a self-contained, specific query that can be understood without the context of the 'chat history'.
        
        Follow these rules strictly:
        1.  Analyze the 'chat history' to understand the context of the conversation.
        2.  Identify any pronouns (e.g., 'it', 'its', 'they', 'that') or vague references in the 'user query'.
        3.  Replace these pronouns and vague references with the specific entities or topics they refer to from the chat history.
        4.  If the 'user query' is already self-contained and specific, return it unchanged.
        5.  CRITICAL: If the 'user query' is about a completely new topic not covered in the chat history, \
            you MUST return it unchanged. Do NOT try to connect it to the previous conversation.
        6.  The rewritten query should be a single, clear question or statement.
        7.  Output ONLY the rewritten query, with no extra text, labels, or explanations.

        Here are some examples of how to behave:

        ---
        Example 1: Rewriting a contextual query
        Chat History:
        User: Do you have the TechPro Ultrabook in stock?
        Assistant: Yes, the TechPro Ultrabook (TP-UB100) is available.
        User query: 'Tell me about its warranty.'
        Rewritten query: 'What is the warranty for the TechPro Ultrabook (TP-UB100)?'
        ---
        Example 2: Handling a topic change
        Chat History:
        User: Do you have the TechPro Ultrabook in stock?
        Assistant: Yes, the TechPro Ultrabook (TP-UB100) is available.
        User query: 'Okay, do you have any monitors?'
        Rewritten query: 'Okay, do you have any monitors?'
        ---
        Example 3: Handling a self-contained query
        Chat History:
        User: What's the price of the BlueWave Gaming Laptop?
        Assistant: The BlueWave Gaming Laptop (BW-GL200) is $1299.99.
        User query: 'What is the price of the GameSphere X console?'
        Rewritten query: 'What is the price of the GameSphere X console?'
        ---

        Chat History:
        {chat_history}
        """

    # Format chat history for the prompt
    #formatted_history = "\n".join([f"User: {user_msg}\nAssistant: {bot_msg}" for user_msg, bot_msg in history])
    formatted_history = ""
    for msg in history:
        if msg["role"] == "user":
            formatted_history += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            formatted_history += f"Assistant: {msg['content']}\n"

    prompt = system_prompt.format(chat_history=formatted_history)
    
    client = get_api_client()
    return client.rewrite_query(query, prompt)
