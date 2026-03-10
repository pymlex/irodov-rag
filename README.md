# Irodov RAG

<img width="1278" height="946" alt="image_2026-03-10_19-31-33" src="https://github.com/user-attachments/assets/fdc6daaf-c6a9-4d2e-a040-e3b00e066a9b" />

A lightweight RAG pipeline for question answering powered with Irodov's 5 volumes on General Physics. The repository is containing a Colab ready notebook that starts an ngrok API for chat.

## Functional abilities

* Download Irodov [dataset](https://www.kaggle.com/datasets/alexzyukov/irodov) from Kaggle. The dataset contains 2.4 million symbols and was prepared with [datalab/marker](https://github.com/pymlex/datalab-pdf2md-example).
* Split documents into chunks and build FAISS index.
* Serve embeddings via a remote embeddings API.
* Query index and compose context aware prompts for an LLM backend.
* Run a FastAPI chat endpoint that returns answer text and source snippets.
* Expose the API via ngrok for remote connections.

---

## Repo base structure

```
.
├── main.ipynb
├── run_server.py
├── requirements.txt
├── data/  # 5 volumes wil be automatically downloaded from Kaggle
├── index/ # FAISS DB index
├── src/
│   ├── build_index.py
│   ├── embeddings_adapter.py
│   ├── hf_llm.py
│   └── server.py
└── README.md
```

---

## Installation

All the installations steps for Colab users are listed in the main.ipynb file. Don't forget to set the following variables: HF_TOKEN, HF_MODEL, EMBEDDINGS_URL, KAGGLE_USERNAME, KAGGLE_KEY, NGROK_AUTH_TOKEN.

---

## API

POST /chat

Request JSON

```json
{
  "session_id": "string",
  "question": "string",
  "k": 6
}
```

Response JSON

```json
{
  "answer": "llm generated answer text",
  "sources": [
    {
      "source": "filename.md",
      "snippet": "first 500 characters of the source"
    }
  ]
}
```

The app is also implemented in Flowise with graph-like interactive interface:

<img width="869" height="663" alt="image_2026-03-10_14-52-44" src="https://github.com/user-attachments/assets/f176c433-4368-45a2-ae5c-84829fc8bd0a" />

---

## Embeddings adapter

Remote embeddings are implemented in src/embeddings_adapter.py

The adapter posts payloads to EMBEDDINGS_URL with shape

```json
{"inputs": ["text 1", "text 2"]}
```

The adapter expects either OpenAI compatible output or a list of embedding objects. The adapter normalizes both shapes into a list of numeric vectors. The project is tested with self-hosted embeddings inference [API](https://github.com/pymlex/embeddings-inference-server).

---

## Example test

Build index and start server. Then test with curl on Windows:

```bash
curl -X POST "https://unsincerely-tiddly-cristian.ngrok-free.dev/chat" -H "Content-Type: application/json" -d "{\"session_id\":\"test\",\"question\":\"Что происходит с лучом света при прохождении через тонкую плёнку?\"}"
```

Expected response contains answer and a list of source snippets.

---

