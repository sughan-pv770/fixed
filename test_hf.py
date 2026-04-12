import requests
import json

hf_token = "hf_" + "XkAswgTQkXMKUqcoBvYHHYhFZCxlnLYyTS"
models = ["BAAI/bge-large-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2", "mixedbread-ai/mxbai-embed-large-v1"]

for model in models:
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": ["Welcome to DocKey AI", "Testing another string"]}
    
    print(f"Testing {model}...")
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            res_json = response.json()
            if isinstance(res_json, list) and len(res_json) > 0:
                print("First output shape approx:", len(res_json), "docs,", len(res_json[0]), "dims")
            else:
                print("Weird format:", str(res_json)[:100])
        else:
            print("Error:", response.text[:100])
    except Exception as e:
        print("Failure:", e)
