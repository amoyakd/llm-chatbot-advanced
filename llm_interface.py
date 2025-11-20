import ollama

def moderate_query(query: str) -> bool:
    """
    Moderates a query using a local llama-guard model via Ollama.

    Args:
        query: The user's query.

    Returns:
        True if the query is safe, False otherwise.
    """
    try:
        # This prompt structure is based on Llama Guard's expected format.
        prompt = f"[INST] Is this prompt safe? '{query}' [/INST]"
        
        response = ollama.generate(
            model="llama-guard3", # Make sure you have a model named 'llamagaurd' in Ollama
            prompt=prompt,
        )
        decision = response["response"].strip()
        
        # Llama Guard's output is typically 'safe' or 'unsafe'.
        if "unsafe" in decision.lower():
            print(f"Query flagged as unsafe: {query}")
            return False
        return True
    except Exception as e:
        print(f"Error during moderation: {e}")
        # Default to safe to avoid blocking users if the moderation model fails.
        return True

def generate_response(query: str, retrieved_docs: list, history: list) -> str:
    """
    Generates a response using a Llama-3.2 model, ensuring it adheres to the retrieved documents.

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
    formatted_history = "\n".join([f"User: {user_msg}\nAssistant: {bot_msg}" for user_msg, bot_msg in history])

    prompt = system_prompt.format(context=context, chat_history=formatted_history)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=messages,
            options={
            'temperature': 0,
            }
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"Error during response generation: {e}")
        return "I'm sorry, but I encountered an error while trying to generate a response."



def rewrite_query(query: str, history: list) -> str:
    """
    Rewrites a conversational query into a self-contained query using the chat history.

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
        5.  CRITICAL: If the 'user query' is about a completely new topic not covered in the chat history, you MUST return it unchanged. Do NOT try to connect it to the previous conversation.
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
    formatted_history = "\n".join([f"User: {user_msg}\nAssistant: {bot_msg}" for user_msg, bot_msg in history])

    prompt = system_prompt.format(chat_history=formatted_history)
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"User query: '{query}'"},
    ]
    
    try:
        response = ollama.chat(
            model="llama3.2", # Using a powerful model for this is a good idea
            messages=messages,
            options={'temperature': 0}
        )
        rewritten = response["message"]["content"].strip()
        # Remove potential quotes around the rewritten query
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]
        if rewritten.startswith("'") and rewritten.endswith("'"):
            rewritten = rewritten[1:-1]
            
        return rewritten
    except Exception as e:
        print(f"Error during query rewrite: {e}")
        return query # Fallback to original query on error
