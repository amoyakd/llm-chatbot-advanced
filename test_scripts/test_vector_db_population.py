import logging
import os
import sys
import shutil
import chromadb
from chromadb.config import Settings

# Add the root directory to the Python path to allow for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_db_manager import run_etl_pipeline
from document_processor import prepare_product_documents, prepare_review_documents

# Configure logging to display info level messages
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Test Configuration ---
# Use a lightweight model for fast testing
TEST_MODEL = 'sentence-transformers/all-MiniLM-L6-v2' 
# Use a temporary directory for the test database
TEST_DB_PATH = "./chroma_db_test" 

def test_vector_db_population():
    """
    Tests the full ETL pipeline: chunking, embedding, and populating ChromaDB.
    It verifies that collections are created and populated correctly.
    """
    logger.info("--- Starting Test: Vector DB Population ---")

    # Define paths to the data files
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    products_file = os.path.join(base_dir, 'products.json')
    reviews_file = os.path.join(base_dir, 'product_reviews.json')
    db_manager = None
    client = None

    # Clean up any previous test database directory
    if os.path.exists(TEST_DB_PATH):
        logger.warning(f"Removing existing test database at: {TEST_DB_PATH}")
        shutil.rmtree(TEST_DB_PATH)

    try:
        # --- 1. Run the ETL Pipeline ---
        logger.info(f"Running ETL pipeline with model '{TEST_MODEL}' into '{TEST_DB_PATH}'")
        db_manager = run_etl_pipeline(
            products_file=products_file,
            reviews_file=reviews_file,
            db_path=TEST_DB_PATH,
            model_name=TEST_MODEL
        )
        logger.info("ETL Pipeline finished.")

        # --- 2. Verification Step ---
        logger.info("\n--- Verifying Database Content ---")
        assert os.path.exists(TEST_DB_PATH), "Database directory was not created."
        
        client = chromadb.PersistentClient(
            path=TEST_DB_PATH,
            settings=Settings(allow_reset=True)
        )
        
        # --- Verify Products Collection ---
        logger.info("[Verifying 'products' collection...]")
        products_collection = client.get_collection(name="products")
        assert products_collection is not None, "'products' collection not found."
        
        # Get expected count from the source
        expected_products_count = len(prepare_product_documents(products_file))
        actual_products_count = products_collection.count()
        assert actual_products_count == expected_products_count, \
            f"Expected {expected_products_count} products, but found {actual_products_count} in DB."
        logger.info(f"SUCCESS: 'products' collection count is correct ({actual_products_count}).")

        # Perform a sample query
        query_result = products_collection.query(query_texts=["lightweight laptop"], n_results=1)
        assert query_result['ids'][0], "Sample query on 'products' collection returned no results."
        logger.info(f"SUCCESS: Sample query on 'products' returned: {query_result['ids'][0][0]}")

        # --- Verify Reviews Collection ---
        logger.info("\n[Verifying 'reviews' collection...]")
        reviews_collection = client.get_collection(name="reviews")
        assert reviews_collection is not None, "'reviews' collection not found."

        # Get expected count from the source
        expected_reviews_count = len(prepare_review_documents(reviews_file, products_file))
        actual_reviews_count = reviews_collection.count()
        assert actual_reviews_count == expected_reviews_count, \
            f"Expected {expected_reviews_count} reviews, but found {actual_reviews_count} in DB."
        logger.info(f"SUCCESS: 'reviews' collection count is correct ({actual_reviews_count}).")

        # Perform a sample query
        query_result = reviews_collection.query(query_texts=["disappointed with battery"], n_results=1)
        assert query_result['ids'][0], "Sample query on 'reviews' collection returned no results."
        logger.info(f"SUCCESS: Sample query on 'reviews' returned: {query_result['ids'][0][0]}")

        # Reset the client to release file locks before cleanup
        logger.info("Shutting down clients to release file locks.")
        if db_manager:
            db_manager.shutdown()
        if client:
            client.reset()

    finally:
        # --- 3. Cleanup Step ---
        logger.info("\n--- Cleaning up test database ---")
        
        # Explicitly shut down both client instances to release file locks
        if db_manager:
            logger.info("Shutting down ETL client...")
            db_manager.shutdown()
        if client:
            logger.info("Shutting down verification client...")
            client.reset()

        # Implement a retry mechanism for shutil.rmtree to handle Windows file lock race conditions
        if os.path.exists(TEST_DB_PATH):
            for i in range(5): # Retry up to 5 times
                try:
                    shutil.rmtree(TEST_DB_PATH)
                    logger.info(f"Successfully removed test database at: {TEST_DB_PATH}")
                    break  # Exit loop if successful
                except (PermissionError, OSError) as e:
                    logger.warning(f"Cleanup attempt {i+1} failed: {e}. Retrying in 2 seconds...")
                    import time
                    time.sleep(2)
            else: # This 'else' belongs to the 'for' loop
                logger.error("Failed to remove test database directory after multiple attempts. It may need to be removed manually.")

    logger.info("\n--- Test Finished Successfully ---")

if __name__ == "__main__":
    test_vector_db_population()
