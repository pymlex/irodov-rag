import os
import requests

EMB_URL = os.environ.get("EMBEDDINGS_URL")

class RemoteEmbeddings:
    def __init__(self):
        self.url = EMB_URL

    def _post(self, payload):
        r = requests.post(self.url, json=payload, timeout=60)
        return r.json()

    def embed_documents(self, texts):
        payload = {"inputs": texts}
        resp = self._post(payload)
        if isinstance(resp, dict) and resp.get("data"):
            out = []
            for item in resp["data"]:
                if isinstance(item, dict) and item.get("embedding") is not None:
                    out.append(item["embedding"])
                else:
                    out.append(item)
            return out
        if isinstance(resp, list):
            return [x.get("embedding") if isinstance(x, dict) and x.get("embedding") is not None else x for x in resp]
        return resp

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def __call__(self, text):
        return self.embed_query(text)