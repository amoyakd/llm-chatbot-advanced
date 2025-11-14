import sys
import os

# Add the project root directory to the Python path to resolve module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

import os
import re
import datetime
import json
from retrieval_manager import RetrievalManager
import llm_interface

# --- Configuration ---
TEST_QUERIES_FILE = "requirements/test_initial_chatbot.md"
OUTPUT_HTML_FILE = "logs/chatbot_test_report.html"

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot Test Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 20px auto; padding: 0 20px; }}
        h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        .test-case {{ background-color: #fdfdfd; border: 1px solid #ecf0f1; border-radius: 8px; margin-bottom: 20px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .query {{ font-weight: bold; color: #2980b9; font-size: 1.2em; }}
        .response {{ background-color: #ecf0f1; border-left: 4px solid #2980b9; padding: 15px; margin-top: 15px; white-space: pre-wrap; font-family: monospace; }}
        .rag-details {{ margin-top: 15px; }}
        .rag-summary {{ cursor: pointer; font-weight: bold; color: #7f8c8d; }}
        .rag-content {{ border: 1px solid #e0e0e0; background-color: #fafafa; padding: 15px; margin-top: 10px; display: none; font-size: 0.9em; max-height: 300px; overflow-y: auto; }}
        .rag-content pre {{ white-space: pre-wrap; word-wrap: break-word; }}
        .retry-section {{ border-top: 2px dashed #f39c12; margin-top: 20px; padding-top: 20px; }}
        .retry-query {{ font-weight: bold; color: #f39c12; }}
        .timestamp {{ font-size: 0.9em; color: #95a5a6; }}
    </style>
</head>
<body>
    <h1>Chatbot Test Report</h1>
    <p class="timestamp">Generated on: {generation_time}</p>
    {test_results}
    <script>
        document.querySelectorAll('.rag-summary').forEach(item => {{
            item.addEventListener('click', event => {{
                const content = item.nextElementSibling;
                if (content.style.display === 'none') {{
                    content.style.display = 'block';
                    item.textContent = '▼ Hide Retrieved Documents';
                }} else {{
                    content.style.display = 'none';
                    item.textContent = '► Show Retrieved Documents';
                }}
            }});
        }});
    </script>
</body>
</html>
"""

def parse_test_queries(file_path):
    """Reads and parses queries from the markdown file."""
    queries = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Match lines that start with a number and a dot, and extract the quoted string
                match = re.search(r'^\d+\.\s*"(.*)"', line)
                if match:
                    queries.append(match.group(1))
    except FileNotFoundError:
        print(f"Error: Test queries file not found at '{file_path}'")
        return []
    return queries

def format_rag_content_as_html(rag_docs):
    """Formats the retrieved documents into a readable HTML string."""
    if not rag_docs:
        return "<p>No documents were retrieved.</p>"
    
    html = ""
    for i, (content, metadata) in enumerate(rag_docs):
        html += f"<h4>Document {i+1}</h4>"
        # Use json.dumps for pretty printing the metadata dictionary
        metadata_str = json.dumps(metadata, indent=2)
        html += f"<pre><strong>Content:</strong>\n{content}\n\n<strong>Metadata:</strong>\n{metadata_str}</pre><hr>"
    return html

def process_query(query, retriever):
    """Retrieves documents and generates a response for a single query."""
    # 1. Retrieve documents using the same logic as the main app
    search_results = retriever.search(query)
    
    retrieved_docs = []
    for collection_name, results in search_results.items():
        if results and results.get('documents') and results['documents'][0]:
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            for i, doc_content in enumerate(docs):
                retrieved_docs.append((doc_content, metadatas[i]))

    # 2. Prepare context for LLM using the same logic as the main app
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
            
            if metadata_parts:
                enhanced_content += "\n" + ", ".join(metadata_parts)
        doc_contents.append(enhanced_content)

    # 3. Generate response
    # Using an empty chat history to ensure each test is isolated
    response = llm_interface.generate_response(query, doc_contents, [])
    
    return response, retrieved_docs

def main():
    """Main function to run the test script."""
    print("Starting chatbot response test...")
    
    # Ensure logs directory exists
    if not os.path.exists("logs"):
        os.makedirs("logs")

    queries = parse_test_queries(TEST_QUERIES_FILE)
    if not queries:
        print("No queries found. Exiting.")
        return
        
    retriever = RetrievalManager()
    
    all_results_html = ""
    
    for i, query in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}: '{query}'")
        
        response, rag_docs = process_query(query, retriever)
        
        result_html = f'''
        <div class="test-case">
            <p class="query">Query: "{query}"</p>
            <div class="rag-details">
                <p class="rag-summary">► Show Retrieved Documents</p>
                <div class="rag-content">{format_rag_content_as_html(rag_docs)}</div>
            </div>
            <div class="response">{response}</div>
        '''
        
        # Check if retry is needed
        if "i'm sorry" in response.lower() or "i am sorry" in response.lower():
            retry_query = f"try harder... {query}"
            print(f"  -> Retrying with: '{retry_query}'")
            
            retry_response, retry_rag_docs = process_query(retry_query, retriever)
            
            result_html += f'''
            <div class="retry-section">
                <p class="retry-query">Retry Query: "{retry_query}"</p>
                <div class="rag-details">
                    <p class="rag-summary">► Show Retrieved Documents (Retry)</p>
                    <div class="rag-content">{format_rag_content_as_html(retry_rag_docs)}</div>
                </div>
                <div class="response">{retry_response}</div>
            </div>
            '''
            
        result_html += "</div>"
        all_results_html += result_html

    # Final HTML content
    final_html = HTML_TEMPLATE.format(
        generation_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        test_results=all_results_html
    )
    
    # Write to file
    with open(OUTPUT_HTML_FILE, 'w', encoding='utf-8') as f:
        f.write(final_html)
        
    print(f"\nTest complete. Report saved to '{os.path.abspath(OUTPUT_HTML_FILE)}'")

if __name__ == "__main__":
    main()