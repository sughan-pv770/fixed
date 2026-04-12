import requests

hf_token = "hf_" + "XkAswgTQkXMKUqcoBvYHHYhFZCxlnLYyTS"
model = "sentence-transformers/all-MiniLM-L6-v2"
url = f"https://router.huggingface.co/hf-inference/models/{model}"
headers = {"Authorization": f"Bearer {hf_token}"}

payloads = [
    {"inputs": "Hello world"},
    {"inputs": ["Hello world", "Test"]},
    {"inputs": ["Hello world"], "options": {"wait_for_model": True}},
]

for p in payloads:
    res = requests.post(url, headers=headers, json=p)
    if res.status_code == 200:
        data = res.json()
        print(f"Success for {type(p['inputs'])}: {len(data)} items")
    else:
        print(f"Failed for {type(p['inputs'])}:", res.status_code, res.text[:200])
