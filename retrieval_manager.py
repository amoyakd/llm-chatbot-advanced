import logging
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os

# Configure basic logging for debugging and tracing
# Use a specific logger name for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class RetrievalManager:
    """
    Manages retrieving documents from a ChromaDB vector database based on a user query.
    """
    def __init__(self, db_path: str = "./chroma_db", model_name: str = 'BAAI/bge-large-en-v1.5'):
        """
        Initializes the RetrievalManager.

        Args:
            db_path (str): Path to the ChromaDB database directory.
            model_name (str): The name of the sentence-transformer model to use for embeddings.
        """
        logger.info(f"Initializing RetrievalManager with db_path='{db_path}' and model='{model_name}'")
        
        # Initialize the ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Load the sentence-transformer model
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformer model '{model_name}'. Error: {e}")
            raise

    def _generate_query_embedding(self, query: str) -> List[float]:
        """
        Generates an embedding for a single query string.

        Args:
            query (str): The user query string.

        Returns:
            List[float]: The embedding vector for the query.
        """
        logger.info(f"Generating embedding for query: '{query}'")
        embedding = self.model.encode(query, convert_to_tensor=False)
        logger.info("Finished generating query embedding.")
        return embedding.tolist()

    def _route_query(self, query: str) -> List[str]:
        """
        Determines which collection(s) to query based on keywords in the query.
        This provides a simple, fast routing mechanism before performing a semantic search.

        Args:
            query (str): The user query string.

        Returns:
            List[str]: A list of collection names to target for the search.
        """
        query_lower = query.lower()
        # Keywords that suggest a user is interested in opinions or experiences
        review_keywords = ["review", "customer", "feedback", "complaints", "say about", "opinion", "experience"]
        # Keywords that suggest a user is interested in product details
        product_keywords = ["specs", "features", "have", "available", "do you have", "specification", "technical details"]

        target_collections = []

        # Check for review-related keywords
        if any(keyword in query_lower for keyword in review_keywords):
            target_collections.append("reviews")
        
        # Check for product-related keywords
        if any(keyword in query_lower for keyword in product_keywords):
            target_collections.append("products")

        # If no specific keywords are found, default to searching both collections
        if not target_collections:
            logger.info("No specific keywords found in query, defaulting to both 'products' and 'reviews' collections.")
            return ["products", "reviews"]
        
        logger.info(f"Routing query to collections: {target_collections}")
        return target_collections

    def search(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Performs a semantic search across relevant collections based on the user query.

        This method orchestrates the query processing, routing, and retrieval steps.

        Args:
            query (str): The user's search query.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary where keys are collection names
                                             and values are the retrieval results from ChromaDB.
        """
        logger.info(f"--- Starting search for query: '{query}' ---")

        # Step 1: Determine which collections to search
        target_collections = self._route_query(query)
        
        # Step 2: Generate an embedding for the query
        query_embedding = self._generate_query_embedding(query)
        
        results = {}

        # Step 3: Query each targeted collection
        for collection_name in target_collections:
            try:
                collection = self.client.get_collection(name=collection_name)
                
                # Set the number of results based on the collection type
                n_results = 5 if collection_name == "products" else 8

                logger.info(f"Querying '{collection_name}' collection with n_results={n_results}...")

                retrieved = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
                results[collection_name] = retrieved
                logger.info(f"Retrieved {len(retrieved.get('ids', [[]])[0])} results from '{collection_name}'.")

            except Exception as e:
                logger.error(f"Failed to query collection '{collection_name}'. Error: {e}")
                results[collection_name] = []
        
        logger.info(f"--- Finished search for query: '{query}' ---")
        return results


if __name__ == '__main__':
    # This allows the script to be run directly to test the retrieval functionality.
    # It assumes the database has been populated by running `vector_db_manager.py` first.
    
    # --- Configuration ---
    EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'
    DB_PATH = "./chroma_db"
    
    # --- Pre-run Check ---
    if not os.path.exists(DB_PATH):
        logger.error(f"FATAL: ChromaDB path '{DB_PATH}' not found.")
        logger.error("Please run 'vector_db_manager.py' first to create and populate the database.")
    else:
        # --- Retrieval Example ---
        logger.info("\n--- Starting Retrieval Test ---")
        try:
            retrieval_manager = RetrievalManager(db_path=DB_PATH, model_name=EMBEDDING_MODEL)

            # Example queries from the retrieval plan
            queries = [
                "What laptops do you have?",
                "Gaming laptops",
                "Lightweight laptop",
                "What do customers say about battery life?",
                "SmartX ProPhone camera reviews",
                "Any feedback on the AudioBliss headphones?"
            ]

            for query in queries:
                print("\n" + "="*60)
                print(f"Executing query: '{query}'")
                print("="*60)
                
                search_results = retrieval_manager.search(query)
                
                for collection_name, results in search_results.items():
                    print(f"\n--- Results from '{collection_name}' collection ---")
                    if results and results.get('documents') and results['documents'][0]:
                        for i, doc in enumerate(results['documents'][0]):
                            distance = results['distances'][0][i] if results.get('distances') else 'N/A'
                            metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                            
                            print(f"  - Result {i+1} (Distance: {distance:.4f}):")
                            print(f"    ID: {results['ids'][0][i]}")
                            print(f"    Document: {doc[:120].strip()}...")
                            print(f"    Metadata: {metadata}")
                    else:
                        print("  No results found in this collection.")
            
            print("\n" + "="*60)
            logger.info("--- Retrieval Test Finished ---")

        except Exception as e:
            logger.error(f"An error occurred during the retrieval test: {e}", exc_info=True)
