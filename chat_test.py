import sqlite3
import requests

conn = sqlite3.connect('dockey.db')
key = conn.execute("SELECT api_key FROM users WHERE username='testuser_hf2'").fetchone()[0]

r = requests.post(
    'http://127.0.0.1:8000/api/chat', 
    json={'query': 'What embeddings model did DocKey migrate to?'}, 
    headers={'Authorization': f'Bearer {key}'}
)
print(r.json())
