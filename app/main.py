from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Tuple
import time
import os

from rag.crawler import crawl
from rag.indexer import build_index
from rag.retriever import Retriever
from rag.generator import generate_answer

app = FastAPI(title="konduit_220962050", version="0.1.1")


class CrawlReq(BaseModel):
    start_url: str
    max_pages: int = Field(30, ge=1, le=200)
    max_depth: int = Field(3, ge=0, le=10)
    crawl_delay_ms: int = Field(500, ge=0, le=5000)

class IndexReq(BaseModel):
    chunk_size: int = Field(800, ge=100, le=4000)
    chunk_overlap: int = Field(100, ge=0, le=2000)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

class AskReq(BaseModel):
    question: str
    top_k: int = Field(5, ge=1, le=20)


@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "rag-mini", "version": app.version}


@app.post("/crawl")
def api_crawl(req: CrawlReq) -> Dict:
    """Crawl pages from a URL, stay in same domain"""
    result = crawl(
        req.start_url,
        req.max_pages,
        req.max_depth,
        req.crawl_delay_ms,
    )
    return result


@app.post("/index")
def api_index(req: IndexReq) -> Dict:
    """Build search index from crawled pages"""
    result = build_index(
        req.chunk_size,
        req.chunk_overlap,
        req.embedding_model,
    )
    return result


@app.post("/ask")
def api_ask(req: AskReq) -> Dict:
    """Answer questions using crawled content"""
    t0 = time.time()
    
    # find relevant chunks
    retriever = Retriever()
    retrievals = retriever.query(req.question, req.top_k)
    t1 = time.time()
    
    # check if results are good enough
    # 0.3 threshold worked well in testing
    top_score = retrievals[0]["score"] if retrievals else 0.0
    try:
        min_score = float(os.getenv("MIN_SCORE", "0.3"))
    except Exception:
        min_score = 0.3
    
    if (not retrievals) or top_score < min_score:
        # not confident enough, say we don't know
        t2 = time.time()
        _log_metrics(t_retrieval_ms=_ms(t0, t1), t_generation_ms=_ms(t1, t2), t_total_ms=_ms(t0, t2))
        return {
            "answer": "not found in crawled content",
            "sources": retrievals[:3],
            "timings": {
                "retrieval_ms": _ms(t0, t1),
                "generation_ms": _ms(t1, t2),
                "total_ms": _ms(t0, t2),
            },
        }
    
    # generate answer
    raw_answer: Union[str, Tuple[str, Optional[Dict]]] = generate_answer(req.question, retrievals)
    
    if isinstance(raw_answer, tuple):
        answer, usage = raw_answer
    else:
        answer, usage = raw_answer, None
    
    # check if LLM refused to answer
    if isinstance(answer, str) and "not found in crawled content" in answer.lower():
        answer = "not found in crawled content"
    
    t2 = time.time()
    
    _log_metrics(t_retrieval_ms=_ms(t0, t1), t_generation_ms=_ms(t1, t2), t_total_ms=_ms(t0, t2))
    
    response: Dict = {
        "answer": answer,
        "sources": [{"url": r["url"], "snippet": r["snippet"]} for r in retrievals],
        "timings": {
            "retrieval_ms": _ms(t0, t1),
            "generation_ms": _ms(t1, t2),
            "total_ms": _ms(t0, t2),
        },
    }
    if usage:
        response["usage"] = usage
    return response


def _ms(t_start: float, t_end: float) -> int:
    return int((t_end - t_start) * 1000)


def _log_metrics(t_retrieval_ms: int, t_generation_ms: int, t_total_ms: int) -> None:
    # save timing data for analysis
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/metrics.csv", "a", encoding="utf-8") as f:
            f.write(f"{t_retrieval_ms},{t_generation_ms},{t_total_ms}\n")
    except Exception:
        pass  # don't crash if logging fails
