import os, json
import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from .utils import cosine_sim


class Retriever:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        model_name = embedding_model.split("/", 1)[-1] if "/" in embedding_model else embedding_model
        self.model = SentenceTransformer(model_name)
        self.vectors = None
        self.meta: List[Dict] = []
    
    def load(self, vec_path="data/vectors.npy", meta_path="data/meta.jsonl"):
        if not os.path.exists(vec_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Index not found. Run /index first.")
        self.vectors = np.load(vec_path)
        self.meta = [json.loads(line) for line in open(meta_path, "r", encoding="utf-8")]
    
    def query(self, question: str, top_k: int = 5) -> List[Dict]:
        """find most relevant chunks"""
        if self.vectors is None:
            self.load()
        
        q_vec = self.model.encode(question)
        sims = cosine_sim(q_vec, self.vectors)
        
        # get top k
        idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        
        return [{"url": self.meta[i]["url"], "snippet": self.meta[i]["text"], "score": float(sims[i])} for i in idxs]
