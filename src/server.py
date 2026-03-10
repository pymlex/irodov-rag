import os
import json
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from .embeddings_adapter import RemoteEmbeddings
from .hf_llm import generate

INDEX_DIR = os.environ.get("INDEX_DIR", "index")
TOP_K = int(os.environ.get("TOP_K", 6))
WINDOW_K = int(os.environ.get("WINDOW_K", 4))

app = FastAPI()
emb = RemoteEmbeddings()
store = FAISS.load_local(
    INDEX_DIR, emb, allow_dangerous_deserialization=True
)

sessions = {}

def get_history(session_id):
    if session_id not in sessions:
        sessions[session_id] = []
    return sessions[session_id]

def add_history(session_id, entry):
    h = get_history(session_id)
    h.append(entry)
    if len(h) > WINDOW_K:
        del h[0]

def build_prompt(question, contexts, history):
    parts = []
    for i, h in enumerate(history):
        parts.append("User: " + h.get("q", "") + "\nAssistant: " + h.get("a", ""))
    parts.append("Use the following contexts to answer the question. If answer is not in the contexts, say you cannot find it.")
    for c in contexts:
        parts.append("SOURCE: " + c.metadata.get("source", "") + "\n" + c.page_content)
    parts.append("Question: " + question)
    prompt = "\n\n".join(parts)
    return prompt

class ChatReq(BaseModel):
    session_id: str
    question: str
    k: int = TOP_K

@app.post("/chat")
async def chat(req: ChatReq):
    docs = store.similarity_search(req.question, k=req.k)
    history = get_history(req.session_id)
    prompt = build_prompt(req.question, docs, history)
    answer = generate(prompt)
    add_history(req.session_id, {"q": req.question, "a": answer})
    sources = []
    for d in docs:
        sources.append({"source": d.metadata.get("source"), "snippet": d.page_content[:500]})
    return {"answer": answer, "sources": sources}