
import sys
import os

# Add the parent directory to the path to find document_processor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from document_processor import DocumentProcessor

# Path to the document to be used for testing
doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'python_lists.txt'))

with open(doc_path, 'r', encoding='utf-8') as f:
    sample_text = f.read()

def test_splitting_methods():
    """Tests the text splitting methods from DocumentProcessor."""
    processor = DocumentProcessor(chunk_size=30, chunk_overlap=10)

    print("--- Input Text ---")
    print(sample_text)
    print("\n" + "="*20 + "\n")

    # Test simple word-based splitting
    print("--- Testing split_text (word-based) ---")
    word_chunks = processor.split_text(sample_text)
    print(f"Found {len(word_chunks)} chunks.\n")
    for i, chunk in enumerate(word_chunks):
        print(f"Chunk {i+1}:")
        print(chunk)
        print("-" * 10)

    print("\n" + "="*20 + "\n")

    # Test sentence-based splitting
    print("--- Testing split_by_sentences ---")
    sentence_chunks = processor.split_by_sentences(sample_text, sentences_per_chunk=2)
    print(f"Found {len(sentence_chunks)} chunks.\n")
    for i, chunk in enumerate(sentence_chunks):
        print(f"Chunk {i+1}:")
        print(chunk)
        print("-" * 10)
        
    print("\n" + "="*20 + "\n")

    # Test recursive splitting
    print("--- Testing recursive_split ---")
    recursive_chunks = processor.recursive_split(sample_text, chunk_size=500)
    print(f"Found {len(recursive_chunks)} chunks.\n")
    for i, chunk in enumerate(recursive_chunks):
        print(f"Chunk {i+1}:")
        print(chunk)
        print("-" * 10)

    print("\n" + "="*20 + "\n")

    # Test smart splitting
    print("--- Testing smart_split ---")
    smart_chunks = processor.smart_split(sample_text, chunk_size=500)
    print(f"Found {len(smart_chunks)} chunks.\n")
    for i, chunk in enumerate(smart_chunks):
        print(f"Chunk {i+1}:")
        print(chunk)
        print("-" * 10)


if __name__ == "__main__":
    test_splitting_methods()
