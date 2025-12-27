import logging
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import os
import json
import re

# Configure basic logging for debugging and tracing
# Use a specific logger name for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class RetrievalManager:
    """
    Manages retrieving documents from a ChromaDB vector database based on a user query.
    """
    # Define a set of common stop words to ignore in category matching
    STOP_WORDS = {
        "a", "an", "the", "is", "are", "and", "or", "but", "for", "with",
        "in", "on", "at", "of", "to", "from", "by", "about", "as", "into",
        "like", "through", "after", "before", "over", "under", "above", "below",
        "up", "down", "out", "off", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
        "don", "should", "now", "compare", "x", "y"
    }
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

        # Load pre-computed filterable metadata
        self.filterable_metadata = None
        try:
            with open("filterable_metadata.json", "r") as f:
                self.filterable_metadata = json.load(f)
            logger.info("Successfully loaded filterable_metadata.json")
        except FileNotFoundError:
            logger.warning("filterable_metadata.json not found. Filtering by brand/category will be disabled.")
        except json.JSONDecodeError:
            logger.error("Failed to decode filterable_metadata.json. Filtering by brand/category will be disabled.")

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

    def _extract_filters(self, query: str) -> Tuple[Dict[str, Any], str]:
        """
        Extracts metadata filters (price, brand, category) from the query string
        and returns a ChromaDB-compatible filter dictionary and a cleaned query.

        Args:
            query (str): The user's raw query.

        Returns:
            A tuple containing the `where` filter dictionary and the cleaned query string.
        """
        cleaned_query = query
        filters = []
        parts_to_remove = []

        # Price filtering
        price_patterns = {
            "lt": re.compile(r"(?:under|less than|below|max of|\$lt)\s*\$?(\d+\.?\d*)", re.IGNORECASE),
            "gt": re.compile(r"(?:over|more than|above|min of|\$gt)\s*\$?(\d+\.?\d*)", re.IGNORECASE),
        }

        for op, pattern in price_patterns.items():
            match = pattern.search(cleaned_query)
            if match:
                price = float(match.group(1))
                filters.append({"price": {f"${op}": price}})
                parts_to_remove.append(match.group(0))

        # Brand and Category filtering
        if self.filterable_metadata:
            # Sort by length descending to match longer names first (e.g., "Computers and Laptops" before "Laptops")
            brands = sorted(self.filterable_metadata.get("brands", []), key=len, reverse=True)
            categories = sorted(self.filterable_metadata.get("categories", []), key=len, reverse=True)

            for brand in brands:
                pattern = r'\b' + re.escape(brand) + r'\b'
                match = re.search(pattern, cleaned_query, re.IGNORECASE)
                if match:
                    filters.append({"brand": brand})
                    # Do not remove brand from query to preserve semantic meaning
                    break  # Assume only one brand filter is needed

            # New, more flexible category matching logic.
            # It checks if any significant word from the query appears in an official category name.
            category_found = False
            query_lower = cleaned_query.lower()

            # 1. Priority check for "laptop", "computer", or "PC"
            if any(term in query_lower for term in ["laptop", "computer", "pc"]):
                # Force category to "Computers and Laptops"
                for category in self.filterable_metadata.get("categories", []):
                    if "Computers and Laptops" in category:
                        filters.append({"category": category})
                        category_found = True
                        break
            
            # 2. If no priority match, use the flexible matching logic
            # if not category_found:
            #     raw_query_words = re.findall(r'\b\w{3,}\b', cleaned_query.lower()) # Extract words of 3+ chars
            #     # Filter out stop words from the query words to prevent irrelevant category matches
            #     query_words = [word for word in raw_query_words if word not in self.STOP_WORDS]

            #     for category in categories:
            #         for word in query_words:
            #             # Check if the query word (handling plurals) is a whole word in the category name.
            #             # e.g., query "cameras" will match category "Cameras and Camcorders".
            #             pattern = r'\b' + re.escape(word.rstrip('s')) + r's?\b'
            #             if re.search(pattern, category, re.IGNORECASE):
            #                 filters.append({"category": category})
            #                 category_found = True
            #                 break  # Word found, move to next category
            #         if category_found:
            #             break  # Category found, stop searching

            # Category search based on synonyms
            # Category search based on synonyms with priority ordering
            CATEGORY_SYNONYMS = {
                "Computers and Laptops": ["laptop", "laptops", "computer", "notebook", "ultrabook", \
                                          "chromebook", "PC", "desktop", "workstation", "gaming laptop"],
                "Cameras and Camcorders": ["camera", "cameras", "camcorder", "photo", "video camera"],
                "Gaming Consoles and Accessories": ["console", "gaming", "games", "controller"],
                "Smartphones and Accessories": ["phone", "smartphone", "mobile", "case", "charger"],
                "Audio Equipment": ["audio", "speaker", "headphone", "earbud", "sound"],
                "Televisions and Home Theater Systems": ["tv", "television", "home theater", "soundbar", "tvs"]
            }

            # Priority list: categories to check first (more specific synonyms)
            priority_categories = ["Computers and Laptops", "Cameras and Camcorders", "Smartphones and Accessories"]
            
            matched_categories = set()
            query_lower = query.lower()

            # Phase 1: Check priority categories for multi-word synonyms first
            for category in priority_categories:
                multi_word_synonyms = [syn for syn in CATEGORY_SYNONYMS[category] if len(syn.split()) > 1]
                for syn in multi_word_synonyms:
                    if syn in query_lower:
                        matched_categories.add(category)
                        break

            # Phase 2: If no priority multi-word matches, check all single-word synonyms
            if not matched_categories:
                for category in CATEGORY_SYNONYMS.keys():
                    single_word_synonyms = [syn for syn in CATEGORY_SYNONYMS[category] if len(syn.split()) == 1]
                    if any(syn in query_lower for syn in single_word_synonyms):
                        matched_categories.add(category)

            # Add all matched categories to filters
            for category in matched_categories:
                filters.append({"category": category})
                category_found = True
        # # [DEACTIVATED] Remove only the identified price-related parts from the query.
        # # This has been deactivated because it weakens the semantic query, leading to less precise results.
        # # The full query context is now preserved for the embedding model.
        # for part in parts_to_remove:
        #     cleaned_query = cleaned_query.replace(part, "", 1)

        # Construct the final filter dictionary
        where_filter = {}
        if filters:
            if len(filters) > 1:
                where_filter = {"$or": filters}
            else:
                where_filter = filters[0]
        
        # # [DEACTIVATED] Final cleanup of extra spaces.
        # cleaned_query = ' '.join(cleaned_query.split()).strip()

        if where_filter:
            logger.info(f"Extracted filter: {where_filter}")
            logger.info(f"Cleaned query for embedding: '{cleaned_query}'")

        return where_filter, cleaned_query

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
        product_keywords = ["specs", "features", "have", "available", "do you have", \
                            "specification", "technical details", "price", "warranty"]

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

        # Step 1: Extract filters and get the cleaned query
        where_filter, cleaned_query = self._extract_filters(query)

        # If cleaning results in an empty string, fall back to the original query for embedding
        embedding_query = cleaned_query if cleaned_query else query
        logger.info(f"Using query for embedding: '{embedding_query}'")

        # Step 2: Determine which collections to search based on the original query's intent
        target_collections = self._route_query(query)
        
        # Step 3: Generate an embedding for the (potentially cleaned) query
        query_embedding = self._generate_query_embedding(embedding_query)
        
        results = {}

        # Step 4: Query each targeted collection with the filter
        for collection_name in target_collections:
            try:
                collection = self.client.get_collection(name=collection_name)
                
                # Set the number of results based on the collection type
                n_results = 5 if collection_name == "products" else 8

                logger.info(f"Querying '{collection_name}' collection with n_results={n_results} and filter={where_filter}...")

                retrieved = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_filter if where_filter else None
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
                "Any feedback on the AudioBliss headphones?",
                "Show me TVs under $500",
                "TechPro gaming laptops over $1000"
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
