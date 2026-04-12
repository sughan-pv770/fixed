from rag import process_and_store_document, answer_query
import chromadb

def test_pipeline():
    print("Testing Fully HF Integrated End-to-End Pipeline...")
    user_id = 111
    doc_id = 111
    
    # Simulating uploading a document describing "Project DocKey"
    file_bytes = b"""
    Project DocKey v2 Launch Notes:
    DocKey has successfully migrated exclusively to Hugging Face embeddings using BAAI/bge-large-en-v1.5.
    The system is much more responsive now.
    """
    filename = "launch_notes.txt"
    try:
        print("1. Injecting Document into Vector DB...")
        process_and_store_document(user_id, doc_id, file_bytes, filename)
        print("Success: Document processed and stored with HF BGE embeddings.")
        
        query = "What embeddings model did DocKey migrate to?"
        print("2. Retrieving chunks and passing to SmolLM...")
        answer = answer_query(user_id, query)
        print("Success: LLM Final Answer ->", answer)
    except Exception as e:
        print("Test failed:", str(e))

if __name__ == "__main__":
    test_pipeline()
