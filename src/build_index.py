import os
import glob
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from .embeddings_adapter import RemoteEmbeddings

DATA_DIR = "data"
INDEX_DIR = os.environ.get("INDEX_DIR", "index")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 2000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))

def load_md_files():
    fps = sorted(glob.glob(os.path.join(DATA_DIR, "*.md")))
    docs = []
    for p in fps:
        with open(p, "r", encoding="utf-8") as f:
            txt = f.read()
            docs.append(Document(page_content=txt, metadata={"source": os.path.basename(p)}))
    return docs

def build_faiss():
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = load_md_files()
    chunks = []
    for d in docs:
        parts = splitter.split_documents([d])
        chunks.extend(parts)
    emb = RemoteEmbeddings()
    store = FAISS.from_documents(chunks, emb)
    os.makedirs(INDEX_DIR, exist_ok=True)
    store.save_local(INDEX_DIR)
    with open(os.path.join(INDEX_DIR, "metadatas.json"), "w", encoding="utf-8") as f:
        json.dump([d.metadata for d in chunks], f, ensure_ascii=False)
    print("saved index to", INDEX_DIR)

if __name__ == "__main__":
    build_faiss()