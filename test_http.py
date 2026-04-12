import requests
import json
import re

URL = "http://127.0.0.1:8000"

print("1. Registration...")
r = requests.post(f"{URL}/register", data={"username": "testuser_hf2", "password": "password123"})
print(r.status_code)

print("2. Login...")
session = requests.Session()
r = session.post(f"{URL}/login", data={"username": "testuser_hf2", "password": "password123"})
print(r.status_code)

print("3. Getting API Key from dashboard...")
r = session.get(f"{URL}/dashboard")
api_key = re.search(r'id="apiKeyInput"\s+value="(dk_[^"]+)"', r.text)
if api_key:
    API_KEY = api_key.group(1)
    print("Found API Key:", API_KEY)
else:
    print("Failed to find API KEY")
    exit()

print("4. Upload Document...")
# Uploading a sample TXT
files = {'file': ('hf_launch.txt', b'Project DocKey v2 Launch Notes:\nDocKey has successfully migrated exclusively to Hugging Face embeddings using BAAI/bge-large-en-v1.5.\nThe system is much more responsive now.', 'text/plain')}
r = session.post(f"{URL}/upload", files=files)
print(r.status_code)

print("5. Chat / RAG...")
# This will trigger embedding generation + retrieval + SmolLM generation
headers = {"Authorization": f"Bearer {API_KEY}"}
r = requests.post(f"{URL}/api/chat", json={"query": "What embeddings model did DocKey migrate to?"}, headers=headers)
print(r.status_code)
try:
    print("Response:", r.json())
except Exception as e:
    print("Error parsing chat response:", r.text[:200])

