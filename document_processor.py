import os
from typing import List, Dict
import re

class DocumentProcessor:
    """Handle document loading and chunking"""
    
    def __init__(self, chunk_size=50, chunk_overlap=10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_text_file(self, filepath: str) -> str:
        """Load text from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_pdf(self, filepath: str) -> str:
        """Load text from PDF"""
        try:
            import pypdf
            text = ""
            with open(filepath, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            print("Please install pypdf: pip install pypdf")
            return ""
    
    def load_directory(self, directory: str) -> Dict[str, str]:
        """Load all supported files from directory"""
        documents = {}
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            if filename.endswith('.txt'):
                documents[filename] = self.load_text_file(filepath)
            elif filename.endswith('.pdf'):
                documents[filename] = self.load_pdf(filepath)
        
        return documents
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap
        This is a simple word-based splitter
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        
        return chunks
    
    def split_by_sentences(self, text: str, sentences_per_chunk=5) -> List[str]:
        """
        Split text by sentences (more semantic)
        Better than word-based for maintaining context
        """
        # Simple sentence splitter (you can improve this)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = '. '.join(sentences[i:i + sentences_per_chunk])
            if chunk:
                chunks.append(chunk + '.')
        
        return chunks
    
    def process_documents(self, directory: str) -> List[Dict]:
        """
        Complete pipeline: load and chunk all documents
        Returns list of chunks with metadata
        """
        documents = self.load_directory(directory)
        all_chunks = []
        
        for filename, content in documents.items():
            chunks = self.split_text(content)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'metadata': {
                        'source': filename,
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    }
                })
        
        return all_chunks
    

    def recursive_split(self, text: str, chunk_size=500, chunk_overlap=50):
        """
        Recursively split text trying different separators
        Maintains hierarchy: paragraphs > lines > sentences > words
        """
        # Separators in order of preference (largest to smallest units)
        separators = [
            "\n\n",   # Paragraphs
            "\n",     # Lines
            ". ",     # Sentences
            "! ",     # Exclamations
            "? ",     # Questions
            "; ",     # Semicolons
            ", ",     # Commas
            " ",      # Words
            ""        # Characters (last resort)
        ]
        
        return self._recursive_split_helper(text, separators, chunk_size, chunk_overlap)

    def _recursive_split_helper(self, text: str, separators: List[str], 
                                chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Helper function for recursive splitting
        """
        final_chunks = []
        
        # Base case: text is small enough
        if len(text) <= chunk_size:
            if text.strip():
                return [text]
            return []
        
        # Get current separator
        separator = separators[0] if separators else ""
        
        # If no separator left, force split by character
        if not separator:
            return self._character_split(text, chunk_size, chunk_overlap)
        
        # Split by current separator
        splits = text.split(separator)
        
        # Reconstruct with separator and build chunks
        current_chunk = ""
        
        for i, split in enumerate(splits):
            # Add separator back (except for first split)
            piece = split if i == 0 else separator + split
            
            # Check if adding this piece exceeds chunk size
            if len(current_chunk) + len(piece) <= chunk_size:
                current_chunk += piece
            else:
                # Save current chunk if not empty
                if current_chunk.strip():
                    final_chunks.append(current_chunk.strip())
                
                # If single piece is too large, use next separator level
                if len(piece) > chunk_size:
                    # Recursively split this piece with next separator
                    sub_chunks = self._recursive_split_helper(
                        piece.strip(), 
                        separators[1:],  # Use next separator
                        chunk_size, 
                        chunk_overlap
                    )
                    final_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    # Start new chunk with this piece
                    current_chunk = piece
        
        # Don't forget the last chunk
        if current_chunk.strip():
            final_chunks.append(current_chunk.strip())
        
        # Add overlap if specified
        if chunk_overlap > 0 and len(final_chunks) > 1:
            final_chunks = self._add_overlap(final_chunks, chunk_overlap)
        
        return final_chunks

    def _character_split(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Force split by character when no separators work
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - chunk_overlap if chunk_overlap > 0 else end
        
        return chunks

    def _add_overlap(self, chunks: List[str], overlap_size: int) -> List[str]:
        """
        Add overlap between chunks for better context preservation
        """
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Get last N characters from previous chunk
                prev_chunk = chunks[i-1]
                overlap = prev_chunk[-overlap_size:] if len(prev_chunk) > overlap_size else prev_chunk
                
                # Add overlap to current chunk
                overlapped_chunks.append(overlap + " " + chunk)
        
        return overlapped_chunks
    


    def smart_split(self, text: str, chunk_size=500, chunk_overlap=50) -> List[str]:
        """
        Smart text splitter that respects sentence boundaries
        Similar to LangChain's RecursiveCharacterTextSplitter but simpler
        
        Args:
            text: Text to split
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
        
        Returns:
            List of text chunks
        """
        import re
        
        # Split into sentences (improved regex)
        sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If single sentence exceeds chunk_size, split it
            if sentence_length > chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence by words
                words = sentence.split()
                temp_chunk = []
                temp_length = 0
                
                for word in words:
                    if temp_length + len(word) + 1 <= chunk_size:
                        temp_chunk.append(word)
                        temp_length += len(word) + 1
                    else:
                        if temp_chunk:
                            chunks.append(' '.join(temp_chunk))
                        temp_chunk = [word]
                        temp_length = len(word)
                
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
                
                continue
            
            # Check if adding this sentence exceeds chunk_size
            if current_length + sentence_length + 1 > chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                if chunk_overlap > 0 and current_chunk:
                    # Include last few sentences for overlap
                    overlap_text = ' '.join(current_chunk)
                    overlap_sentences = []
                    overlap_length = 0
                    
                    # Add sentences from the end until we reach overlap size
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) <= chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_length += len(s)
                        else:
                            break
                    
                    current_chunk = overlap_sentences + [sentence]
                    current_length = sum(len(s) for s in current_chunk) + len(current_chunk)
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_length + 1
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


# Test it
if __name__ == "__main__":
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
    
    # Test with a sample text
    sample_text = """
    Python is a high-level programming language. It was created by Guido van Rossum.
    Python emphasizes code readability. It uses significant whitespace.
    Python supports multiple programming paradigms. These include object-oriented and functional programming.
    Python has a large standard library. It is often described as a "batteries included" language.
    """
    
    chunks = processor.split_text(sample_text)
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk[:100] + "...")

# =====================================================================================
# New additions for RAG plan implementation (Chunking)
# =====================================================================================

import json
import logging
from typing import List, Dict, Any

# Configure basic logging for debugging and tracing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_product_documents(products_file_path: str) -> List[Dict[str, Any]]:
    """
    Loads product data from a JSON file and prepares it for embedding,
    treating each product as a single document.

    Args:
        products_file_path (str): The path to the products JSON file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                              represents a product document with its ID,
                              text for embedding, and metadata.
    """
    logging.info(f"Starting to process product documents from: {products_file_path}")
    documents = []
    try:
        with open(products_file_path, 'r', encoding='utf-8') as f:
            products_data = json.load(f)

        for product_name, product in products_data.items():
            # 1. Generate a Unique ID
            unique_id = f"product-{product['model_number']}"

            # 2. Prepare Text for Embedding
            features_text = ', '.join(product.get('features', []))
            text_for_embedding = (
                f"Product: {product.get('name', '')}, "
                f"Brand: {product.get('brand', '')}, "
                f"Category: {product.get('category', '')}. "
                f"Features: {features_text}. "
                f"Description: {product.get('description', '')}"
            )

            # 3. Prepare Metadata
            metadata = {
                "chunk_type": "product_info",
                "product_name": product.get('name'),
                "model_number": product.get('model_number'),
                "category": product.get('category'),
                "brand": product.get('brand'),
                "price": product.get('price')
            }
            
            # Remove None values from metadata
            metadata = {k: v for k, v in metadata.items() if v is not None}

            documents.append({
                "id": unique_id,
                "text_for_embedding": text_for_embedding,
                "metadata": metadata
            })
        
        logging.info(f"Successfully prepared {len(documents)} product documents.")

    except FileNotFoundError:
        logging.error(f"File not found: {products_file_path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {products_file_path}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred in prepare_product_documents: {e}")
        return []
        
    return documents

def prepare_review_documents(reviews_file_path: str, products_file_path: str) -> List[Dict[str, Any]]:
    """
    Loads review data, enriches it with product metadata, and prepares it for
    embedding, treating each review as a single chunk.

    Args:
        reviews_file_path (str): The path to the product reviews JSON file.
        products_file_path (str): The path to the products JSON file for metadata enrichment.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                              represents a review chunk with its ID,
                              text for embedding, and metadata.
    """
    logging.info(f"Starting to process review documents from: {reviews_file_path}")
    documents = []

    try:
        # Load product data for metadata enrichment
        with open(products_file_path, 'r', encoding='utf-8') as f:
            products_data = json.load(f)
        
        # Create a lookup map from model_number to product details
        product_lookup = {
            details['model_number']: {
                "category": details.get('category'),
                "brand": details.get('brand'),
                "product_name": details.get('name')
            } for _, details in products_data.items()
        }

        with open(reviews_file_path, 'r', encoding='utf-8') as f:
            reviews_data = json.load(f)

        # Correctly iterate over the list of review groups
        for product_review_group in reviews_data:
            model_number = product_review_group.get('model_number')
            if not model_number:
                logging.warning("Found a review group with no model_number. Skipping.")
                continue

            product_info = product_lookup.get(model_number)
            if not product_info:
                logging.warning(f"Could not find product metadata for model_number: {model_number}. Skipping reviews.")
                continue

            for review in product_review_group.get('reviews', []):
                # 1. Generate a Unique ID
                # The key for review text is 'review', not 'review_text'
                review_text = review.get('review', '')
                review_id = review.get('review_id')
                if not review_id:
                    logging.warning("Found a review with no review_id. Skipping.")
                    continue
                unique_id = f"review-{review_id}"


                # 2. Prepare Text for Embedding
                text_for_embedding = review_text

                # 3. Prepare Metadata
                metadata = {
                    "chunk_type": "review",
                    "product_name": product_info['product_name'],
                    "model_number": model_number,
                    "category": product_info['category'],
                    "brand": product_info['brand'],
                    "rating": review.get('rating')
                }
                
                # Remove None values from metadata
                metadata = {k: v for k, v in metadata.items() if v is not None}

                documents.append({
                    "id": unique_id,
                    "text_for_embedding": text_for_embedding,
                    "metadata": metadata
                })

        logging.info(f"Successfully prepared {len(documents)} review documents.")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e.filename}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from a file: {e}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred in prepare_review_documents: {e}", exc_info=True)
        return []

    return documents