import json
import urllib.request

HF_API_KEY = "hf_" + "XkAswgTQkXMKUqcoBvYHHYhFZCxlnLYyTS"
HF_EMBED_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5"

def custom_ef(input):
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": input,
        "options": {"wait_for_model": True}
    }
    req = urllib.request.Request(
        HF_EMBED_URL, 
        data=json.dumps(payload).encode("utf-8"), 
        headers=headers
    )
    with urllib.request.urlopen(req) as f:
        res = json.loads(f.read().decode("utf-8"))
        print("\n=== DEBUG ===")
        print("Type res:", type(res))
        print("Len res:", len(res))
        if len(res) > 0:
            print("Type res[0]:", type(res[0]))
            
            if type(res[0]) == list:
                print("Len res[0]:", len(res[0]))
                if len(res[0]) > 0:
                    print("Type res[0][0]:", type(res[0][0]))
                    if type(res[0][0]) == list:
                        print("Len res[0][0]:", len(res[0][0]))
        print("=============\n")
        
    return res

out = custom_ef(["Project DocKey v2 Launch Notes... \n ..."])
