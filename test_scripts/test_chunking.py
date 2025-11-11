import logging
import os
import sys
from pprint import pprint

# Add the root directory to the Python path to allow for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from document_processor import prepare_product_documents, prepare_review_documents

# Configure logging to display info level messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_chunking_implementation():
    """
    Tests the document preparation functions for products and reviews,
    verifying their output structure and content as per the plan.
    """
    logging.info("--- Starting Test: Chunking and Document Preparation ---")

    # Define paths to the data files, assuming the script is run from the project root
    # or the test_scripts directory.
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    products_file = os.path.join(base_dir, 'products.json')
    reviews_file = os.path.join(base_dir, 'product_reviews.json')
    
    logging.info(f"Using products file: {products_file}")
    logging.info(f"Using reviews file: {reviews_file}")

    # --- Test 1: Product Document Preparation ---
    logging.info("\n[1] Testing Product Document Preparation...")
    product_documents = prepare_product_documents(products_file)

    # Assert that documents were created
    assert product_documents is not None, "prepare_product_documents returned None"
    assert isinstance(product_documents, list), "Expected a list of product documents"
    assert len(product_documents) > 0, "No product documents were created"
    logging.info(f"SUCCESS: Created {len(product_documents)} product documents.")

    # Verify the structure of the first product document
    first_product = product_documents[0]
    expected_keys = ["id", "text_for_embedding", "metadata"]
    assert all(key in first_product for key in expected_keys), \
        f"Product document is missing one of the expected keys: {expected_keys}"
    
    expected_metadata_keys = ["chunk_type", "product_name", "model_number", "category", "brand", "price"]
    assert all(key in first_product["metadata"] for key in expected_metadata_keys), \
        f"Product metadata is missing one of the expected keys: {expected_metadata_keys}"

    logging.info("SUCCESS: First product document has the correct structure.")
    
    # Print the first product for visual inspection
    print("\n--- Sample Product Document ---")
    pprint(first_product)
    print("-----------------------------\n")


    # --- Test 2: Review Document Preparation ---
    logging.info("[2] Testing Review Document Preparation...")
    review_documents = prepare_review_documents(reviews_file, products_file)

    # Assert that documents were created
    assert review_documents is not None, "prepare_review_documents returned None"
    assert isinstance(review_documents, list), "Expected a list of review documents"
    assert len(review_documents) > 0, "No review documents were created"
    logging.info(f"SUCCESS: Created {len(review_documents)} review documents.")

    # Verify the structure of the first review document
    first_review = review_documents[0]
    assert all(key in first_review for key in expected_keys), \
        f"Review document is missing one of the expected keys: {expected_keys}"

    expected_metadata_keys = ["chunk_type", "product_name", "model_number", "category", "brand", "rating"]
    assert all(key in first_review["metadata"] for key in expected_metadata_keys), \
        f"Review metadata is missing one of the expected keys: {expected_metadata_keys}"

    logging.info("SUCCESS: First review document has the correct structure.")

    # Print the first review for visual inspection
    print("\n--- Sample Review Document ---")
    pprint(first_review)
    print("----------------------------\n")
    
    logging.info("--- Test Finished Successfully ---")

if __name__ == "__main__":
    test_chunking_implementation()
