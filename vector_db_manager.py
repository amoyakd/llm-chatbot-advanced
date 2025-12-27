import logging
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os

# Ensure the document_processor functions are importable
from document_processor import prepare_product_documents, prepare_review_documents

# Configure basic logging for debugging and tracing
# Use a specific logger name for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class VectorDBManager:
    """
    Manages the creation, embedding, and population of a ChromaDB vector database.
    """
    def __init__(self, db_path: str = "./chroma_db", model_name: str = 'BAAI/bge-large-en-v1.5'):
        """
        Initializes the VectorDBManager.

        Args:
            db_path (str): Path to the ChromaDB database directory.
            model_name (str): The name of the sentence-transformer model to use for embeddings.
        """
        logger.info(f"Initializing VectorDBManager with db_path='{db_path}' and model='{model_name}'")
        
        # Initialize the ChromaDB client, allowing reset for cleanup operations
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(allow_reset=True)
        )
        
        # Load the sentence-transformer model
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformer model '{model_name}'. Error: {e}")
            raise

    def shutdown(self):
        """
        Shuts down the manager and resets the ChromaDB client to release file locks.
        """
        logger.info("Shutting down VectorDBManager and resetting client.")
        self.client.reset()

    def _generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generates embeddings for a list of texts in batches.

        Args:
            texts (List[str]): A list of text strings to embed.
            batch_size (int): The batch size for embedding generation.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        logger.info(f"Generating embeddings for {len(texts)} documents in batches of {batch_size}...")
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        logger.info("Finished generating embeddings.")
        return embeddings.tolist()

    def populate_collection(self, collection_name: str, documents: List[Dict[str, Any]]):
        """
        Populates a ChromaDB collection with documents, generating embeddings first.

        Args:
            collection_name (str): The name of the collection to create/populate.
            documents (List[Dict[str, Any]]): A list of prepared documents from document_processor.
        """
        if not documents:
            logger.warning(f"No documents provided for collection '{collection_name}'. Skipping population.")
            return

        logger.info(f"Populating collection: '{collection_name}'")

        # Delete the collection if it already exists to ensure a fresh start
        try:
            if collection_name in [c.name for c in self.client.list_collections()]:
                logger.warning(f"Collection '{collection_name}' already exists. Deleting it for a fresh population.")
                self.client.delete_collection(name=collection_name)
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {e}")
            # Continue anyway, maybe creation will fail with a clearer error
        
        # Create the collection
        collection = self.client.create_collection(name=collection_name)

        # Unpack documents into separate lists for ChromaDB
        ids = [doc['id'] for doc in documents]
        texts_to_embed = [doc['text_for_embedding'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]

        # Generate embeddings
        embeddings = self._generate_embeddings(texts_to_embed)

        # Add data to the collection in batches to avoid potential client-side limits
        batch_size = 500 
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            # The 'documents' parameter in ChromaDB stores the original text
            batch_documents_text = texts_to_embed[i:i + batch_size]

            try:
                logger.info(f"Adding batch {i//batch_size + 1} to '{collection_name}' collection...")
                collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_documents_text,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            except Exception as e:
                logger.error(f"Failed to add batch to collection '{collection_name}'. Error: {e}")
                # Optionally, decide if you want to stop or continue on error
                return

        logger.info(f"Successfully populated '{collection_name}' with {collection.count()} items.")

def run_etl_pipeline(products_file: str, reviews_file: str, db_path: str, model_name: str):
    """
    Runs the full ETL (Extract, Transform, Load) pipeline to populate the vector database.
    
    Args:
        products_file (str): Path to the products JSON file.
        reviews_file (str): Path to the product reviews JSON file.
        db_path (str): Path to store the ChromaDB database.
        model_name (str): Name of the sentence-transformer model.
    """
    logger.info("--- Starting RAG ETL Pipeline ---")
    
    # Initialize the manager
    db_manager = VectorDBManager(db_path=db_path, model_name=model_name)

    # --- Process and Populate Products ---
    logger.info("Step 1: Preparing product documents...")
    product_documents = prepare_product_documents(products_file)
    db_manager.populate_collection("products", product_documents)

    # --- Process and Populate Reviews ---
    logger.info("Step 2: Preparing review documents...")
    review_documents = prepare_review_documents(reviews_file, products_file)
    db_manager.populate_collection("reviews", review_documents)
    
    logger.info("--- RAG ETL Pipeline Finished Successfully ---")
    return db_manager


if __name__ == '__main__':
    # This allows the script to be run directly to populate the database.
    # It assumes the script is run from the root of the project directory.
    
    # Use the recommended model for production/main runs
    EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'
    DB_PATH = "./chroma_db"
    
    # Define file paths
    CWD = os.getcwd()
    PRODUCTS_JSON_PATH = os.path.join(CWD, 'products.json')
    REVIEWS_JSON_PATH = os.path.join(CWD, 'product_reviews.json')

    # Check if files exist
    if not os.path.exists(PRODUCTS_JSON_PATH) or not os.path.exists(REVIEWS_JSON_PATH):
        logger.error("Error: Make sure 'products.json' and 'product_reviews.json' exist in the project root directory.")
    else:
        run_etl_pipeline(
            products_file=PRODUCTS_JSON_PATH,
            reviews_file=REVIEWS_JSON_PATH,
            db_path=DB_PATH,
            model_name=EMBEDDING_MODEL
        )
