import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

class MyEF(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return [[0.0]*1024]

c = chromadb.PersistentClient('chroma_test_9')
cef = MyEF()
c.get_or_create_collection('test', embedding_function=cef)
print("SUCCESS")
