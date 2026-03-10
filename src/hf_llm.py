import os
import requests

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_MODEL = os.environ.get("HF_MODEL")
API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

def generate(prompt, max_new_tokens=512):
    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "stream": False
    }
    r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
    j = r.json()
    if "choices" in j and len(j["choices"]) > 0 and "message" in j["choices"][0]:
        return j["choices"][0]["message"]["content"]
    return str(j)