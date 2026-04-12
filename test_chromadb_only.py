import chromadb
import numpy as np

# Let's bypass HF embedding and just insert a vector of size 9 and size 1024 directly!
print("Starting...")

CHROMA_PATH = "chroma_data_v5"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    # Test 1024 dimension
    col = chroma_client.get_or_create_collection(name="test_1024")
    col.add(ids=["1"], embeddings=[[1.0]*1024], documents=["test document 1024"])
    print("Successfully added 1024-dim embedding.")
except Exception as e:
    print("Error 1024:", e)

try:
    # Test 9 dimension
    col2 = chroma_client.get_or_create_collection(name="test_9")
    col2.add(ids=["1"], embeddings=[[1.0]*9], documents=["test document 9"])
    print("Successfully added 9-dim embedding.")
except Exception as e:
    print("Error 9:", e)
