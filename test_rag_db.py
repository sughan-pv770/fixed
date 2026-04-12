import requests
import sqlite3

URL = "http://127.0.0.1:8000"
username = "testuser_hf2"

# Get password logic doesn't matter if we just bypass auth for the UI using the previous session
print("Logging in...")
session = requests.Session()
r = session.post(f"{URL}/login", data={"username": username, "password": "password123"})

print("Uploading to v7...")
files = {'file': ('hf_launch.txt', b'Project DocKey v2 Launch Notes:\nDocKey has successfully migrated exclusively to Hugging Face embeddings using BAAI/bge-large-en-v1.5.\nThe system is much more responsive now.', 'text/plain')}
r = session.post(f"{URL}/upload", files=files)
print("Upload status:", r.status_code)

print("Retrieving API key for API call...")
conn = sqlite3.connect('dockey.db')
key = conn.execute(f"SELECT api_key FROM users WHERE username='{username}'").fetchone()[0]

print("Chatting...")
r = requests.post(
    f'{URL}/api/chat', 
    json={'query': 'What embeddings model did DocKey migrate to?'}, 
    headers={'Authorization': f'Bearer {key}'}
)
print("Chat Response:", r.json())
