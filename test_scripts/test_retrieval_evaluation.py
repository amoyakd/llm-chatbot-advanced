import logging
import os
import sys
from datetime import datetime

# Adjust the path to import from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval_manager import RetrievalManager

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Test Configuration ---
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'
REPORT_DIR = "./logs"
REPORT_FILE = os.path.join(REPORT_DIR, f"retrieval_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")

# The test set of queries from the retrieval plan
EVALUATION_QUERIES = [
    {
        "id": 1,
        "query": "What laptops do you have?",
        "expected_collections": ["products"],
        "notes": "Should return a variety of laptops from the products collection."
    },
    {
        "id": 2,
        "query": "Do you have any Gaming laptops?",
        "expected_collections": ["products"],
        "notes": "Should return laptops with 'gaming' in their description or specs."
    },
    {
        "id": 3,
        "query": "What Lightweight laptops do you have",
        "expected_collections": ["products"],
        "notes": "Pure semantic search. Should find laptops described as lightweight, portable, etc."
    },
    {
        "id": 4,
        "query": "Budget camera under $300",
        "expected_collections": ["products"],
        "notes": "This version of retrieval doesn't filter by price, so we expect semantic results for 'budget camera'."
    },
    {
        "id": 5,
        "query": "Share more details on SmartX ProPhone camera reviews",
        "expected_collections": ["reviews"],
        "notes": "Should retrieve reviews specifically for the 'SmartX ProPhone'."
    },
    {
        "id": 6,
        "query": "What do customers say about battery life of TechPro Ultrabook?",
        "expected_collections": ["reviews"],
        "notes": "Semantic search on reviews. Should find reviews mentioning battery about TechPro Ultrabook."
    },
    {
        "id": 7,
        "query": "What TV under $500 do you have?",
        "expected_collections": ["products"],
        "notes": "No price filtering, so should return TVs based on semantic search."
    },
    {
        "id": 8,
        "query": "What Audio products do you have",
        "expected_collections": ["products"],
        "notes": "Should retrieve products from the 'Audio' category."
    },
    {
        "id": 9,
        "query": "Customer complaints about Ultrabook",
        "expected_collections": ["reviews"],
        "notes": "Should find negative reviews (complaints) for products named 'Ultrabook'."
    },
    {
        "id": 10,
        "query": "Compare GameSphere X and Y",
        "expected_collections": ["products", "reviews"],
        "notes": "Should retrieve specs for both products and potentially reviews comparing them."
    }
]

def generate_report_header():
    """Generates the header for the HTML report."""
    header = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retrieval Evaluation Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 2em; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 1em; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        ul {{ margin: 0; padding-left: 20px; }}
    </style>
</head>
<body>
    <h1>Retrieval Evaluation Report</h1>
    <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Database Path:</strong> <code>{DB_PATH}</code></p>
    <p><strong>Embedding Model:</strong> <code>{EMBEDDING_MODEL}</code></p>
    
    <h2>How to Interpret the 'Dist.' (Distance) Value</h2>
    <p>The <code>Dist.</code> value represents the dissimilarity between the query and the retrieved document. A <strong>lower value is better</strong>, indicating higher semantic relevance.</p>

    <table>
        <thead>
            <tr>
                <th>Query ID</th>
                <th>Query</th>
                <th>Expected</th>
                <th>Retrieved</th>
                <th>Pass/Fail</th>
                <th>Notes</th>
            </tr>
        </thead>
        <tbody>
"""
    return header

def generate_report_footer():
    """Generates the footer for the HTML report."""
    return """
        </tbody>
    </table>
</body>
</html>
"""

def run_evaluation():
    """
    Runs the full evaluation process: executes queries, prints results to console,
    and generates an HTML report.
    """
    logger.info("--- Starting Retrieval Evaluation Script ---")

    # --- Pre-run Check ---
    if not os.path.exists(DB_PATH):
        logger.error(f"FATAL: ChromaDB path '{DB_PATH}' not found.")
        logger.error("Please run 'vector_db_manager.py' first to create and populate the database.")
        return

    # --- Initialize Manager ---
    try:
        retrieval_manager = RetrievalManager(db_path=DB_PATH, model_name=EMBEDDING_MODEL)
    except Exception as e:
        logger.error(f"Failed to initialize RetrievalManager: {e}", exc_info=True)
        return

    # --- Prepare Report ---
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_content = generate_report_header()

    # --- Execute Queries ---
    for item in EVALUATION_QUERIES:
        query_id = item["id"]
        query = item["query"]
        
        print("\n" + "="*80)
        logger.info(f"Executing Query #{query_id}: '{query}'")
        print("="*80)

        search_results = retrieval_manager.search(query)
        
        retrieved_summary = []

        for collection_name, results in search_results.items():
            print(f"\n--- Results from '{collection_name}' collection ---")
            if results and results.get('documents') and results['documents'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    doc_text = results['documents'][0][i]
                    distance = results['distances'][0][i]
                    metadata = results['metadatas'][0][i]
                    
                    display_text = ""
                    if collection_name == "products":
                        product_name = metadata.get("product_name", "N/A")
                        display_text = f"Product: {product_name}"
                    elif collection_name == "reviews":
                        # Take first 15 words of the review text
                        words = doc_text.split()
                        display_text = "Review: " + " ".join(words[:15]) + ("..." if len(words) > 15 else "")
                    
                    summary_item = f"<li>{collection_name}: {doc_id} - {display_text} (Dist: {distance:.4f})</li>"
                    retrieved_summary.append(summary_item)

                    print(f"  - Result {i+1} (ID: {doc_id}, Distance: {distance:.4f})")
                    print(f"    Type: {collection_name}")
                    print(f"    Display Text: {display_text}")
                    print(f"    Metadata: {metadata}")
                    print(f"    Document: {doc_text[:150].strip()}...")
            else:
                print("  No results found in this collection.")
        
        # Append to HTML report
        retrieved_html = f"<ul>{''.join(retrieved_summary)}</ul>" if retrieved_summary else "None"
        report_content += f"""
            <tr>
                <td>{query_id}</td>
                <td>{query}</td>
                <td>{' & '.join(item['expected_collections'])}</td>
                <td>{retrieved_html}</td>
                <td></td>
                <td>{item['notes']}</td>
            </tr>
"""

    report_content += generate_report_footer()

    # --- Save Report ---
    try:
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info(f"Successfully generated evaluation report: {REPORT_FILE}")
    except IOError as e:
        logger.error(f"Failed to write report file: {e}")

    logger.info("--- Retrieval Evaluation Script Finished ---")


if __name__ == '__main__':
    run_evaluation()
