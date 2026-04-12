import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import sys

print("STEP 1: Import json, urllib")
import json
import urllib.request
import urllib.error

print("STEP 2: Import chromadb")
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

print("STEP 3: Import pypdf")
from pypdf import PdfReader
from io import BytesIO

print("STEP 4: Declare HF URLs")
HF_API_KEY = "hf_" + "XkAswgTQkXMKUqcoBvYHHYhFZCxlnLYyTS"
HF_EMBED_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5"
LLM_CHAT_URL = "https://Parasuramane24-smollm2-fast-api.hf.space/chat"

print("STEP 5: Define class")
class HuggingFaceCustomEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        pass

print("STEP 6: Chroma Client")
CHROMA_PATH = "chroma_data_v4"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

print("STEP 7: Init custom embed")
custom_ef = HuggingFaceCustomEmbeddingFunction()

print("STEP 8: get or create collection!")
collection = chroma_client.get_or_create_collection(
    name="dockey_docs_hf_bge_new3",
    embedding_function=custom_ef
)

print("SUCCESS: ALL EXECUTED")
