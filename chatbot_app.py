import gradio as gr
from retrieval_manager import RetrievalManager
import llm_interface
import json
import logging
import os
from vector_db_manager import run_etl_pipeline

from dotenv import load_dotenv
load_dotenv()  # Loads .env file automatically


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Database and Model Configuration ---
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'
PRODUCTS_JSON_PATH = 'products.json'
REVIEWS_JSON_PATH = 'product_reviews.json'

# 1. Instantiate the retrieval manager
# It will now connect to the newly created or existing database
retriever = RetrievalManager(db_path=DB_PATH, model_name=EMBEDDING_MODEL)

def respond(message, chat_history):
    """
    Main response function for the Gradio chatbot.

    Orchestrates moderation, retrieval, and response generation.
    Leverages Gradio's built-in chat history management.

    Args:
        message: The user's input message.
        chat_history: The history of the conversation, managed by Gradio.

    Returns:
        A tuple containing:
        - An empty string to clear the input textbox.
        - The updated chat history.
        - The retrieved documents formatted for JSON display.
    """
    # 2. Moderate the user's query
    if not llm_interface.moderate_query(message):
        response = "I'm sorry, but your query violates our safety guidelines. I cannot process this request."
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": response})
        return "", chat_history, []

    # 3. Rewrite the query for context
    rewritten_query = llm_interface.rewrite_query(message, chat_history)
    logger.info(f"Original query: '{message}' | Rewritten query: '{rewritten_query}'")

    # 4. Retrieve relevant documents
    search_results = retriever.search(rewritten_query)
    
    # Process the search results
    retrieved_docs = []
    for collection_name, results in search_results.items():
        if results and results.get('documents') and results['documents'][0]:
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            for i, doc_content in enumerate(docs):
                retrieved_docs.append((doc_content, metadatas[i]))

    # Incorporate metadata into content for LLM
    doc_contents = []
    for content, metadata in retrieved_docs:
        enhanced_content = content
        
        if metadata:
            metadata_parts = []
            if 'product_name' in metadata and metadata['product_name'] not in enhanced_content:
                metadata_parts.append(f"Product Name: {metadata['product_name']}")
            if 'brand' in metadata and metadata['brand'] not in enhanced_content:
                metadata_parts.append(f"Brand: {metadata['brand']}")
            if 'category' in metadata and metadata['category'] not in enhanced_content:
                metadata_parts.append(f"Category: {metadata['category']}")
            if 'price' in metadata:
                metadata_parts.append(f"Price: ${metadata['price']:.2f}")
            if 'rating' in metadata:
                metadata_parts.append(f"Rating: {metadata['rating']} out of 5")
            if 'warranty' in metadata:
                metadata_parts.append(f"Warranty: {metadata['warranty']}")
            
            if metadata_parts:
                enhanced_content += "\n" + ", ".join(metadata_parts)
        doc_contents.append(enhanced_content)

    # 5. Generate a response using the LLM
    response = llm_interface.generate_response(message, doc_contents, chat_history)
    
    # 6. Append messages in Gradio 6.0 format
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})
    
    # 7. Return values to update the Gradio UI
    docs_for_display = [
        {"content": content, "metadata": metadata} for content, metadata in retrieved_docs
    ]
    
    return "", chat_history, docs_for_display

# --- Gradio UI Definition ---
with gr.Blocks() as demo:
    gr.Markdown("# 🛍️ Product Inquiry Chatbot")
    gr.Markdown("Ask me anything about our products and I will do my best to answer based on the information I have.")
    
    chatbot = gr.Chatbot(
        height=550,
        show_label=False,
        avatar_images=("static/images/user.png", "static/images/bot.png")
        #bubble_full_width=False,
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="e.g., 'What kind of laptops do you have?'",
            show_label=False,
            scale=4,
            container=False
        )
        submit = gr.Button("Send", scale=1, variant="primary")
    
    with gr.Accordion("📄 Retrieved Documents", open=False):
        docs_display = gr.JSON(label="Source Documents Used for Response")
    
    gr.Examples(
        examples=[
            "What laptops do you have?",
            "Compare the GameSphere X and Y consoles.",
            "What do customers say about the battery life of the InnovateBook?",
            "Is there a budget-friendly camera under $300?"
        ],
        inputs=msg,
        label="Example Questions"
    )
    
    # Connect the respond function to the UI events
    submit.click(respond, [msg, chatbot], [msg, chatbot, docs_display])
    msg.submit(respond, [msg, chatbot], [msg, chatbot, docs_display])

if __name__ == "__main__":
    print("Starting Gradio app... Access it at http://127.0.0.1:7860")
    demo.launch(theme=gr.themes.Soft(primary_hue="slate", secondary_hue="blue"))
