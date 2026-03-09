import os, json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List
from .utils import chunks, normalize_ws


def load_pages(path="data/pages.jsonl") -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run /crawl first.")
    pages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))
    return pages


def build_index(chunk_size: int = 800, chunk_overlap: int = 100, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict:
    """chunk pages and create embeddings"""
    pages = load_pages()
    
    # load embedding model
    model_name = embedding_model.split("/", 1)[-1] if "/" in embedding_model else embedding_model
    model = SentenceTransformer(model_name)
    
    vectors = []
    meta = []
    
    for p in pages:
        url = p["url"]
        text = p["text"]
        
        # break into chunks with overlap
        for i, ch in enumerate(chunks(text, chunk_size, chunk_overlap)):
            ch_norm = normalize_ws(ch)
            
            if len(ch_norm) < 50:  # skip tiny chunks
                continue
            
            meta.append({"url": url, "chunk_id": i, "text": ch_norm[:1000]})
            vectors.append(model.encode(ch_norm))
    
    if not vectors:
        raise RuntimeError("No chunks produced; cannot build index.")
    
    # save vectors and metadata
    os.makedirs("data", exist_ok=True)
    np_vectors = np.vstack(vectors)
    np.save("data/vectors.npy", np_vectors)
    
    with open("data/meta.jsonl","w",encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False)+"\n")
    
    return {"vector_count": int(np_vectors.shape[0]), "errors": []}

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk_size", type=int, default=800)
    ap.add_argument("--chunk_overlap", type=int, default=100)
    ap.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    print(json.dumps(build_index(args.chunk_size, args.chunk_overlap, args.embedding_model), indent=2))
