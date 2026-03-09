# helper functions

import re
import time
import hashlib
from typing import Iterable


def now_ms() -> int:
    return int(time.time() * 1000)


def normalize_ws(text: str) -> str:
    """collapse whitespace"""
    return re.sub(r"\s+", " ", text or "").strip()


def canonical_url(url: str) -> str:
    """remove trailing slashes"""
    import re as _re
    return _re.sub(r"/+$", "", url.strip())


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def chunks(text: str, size: int, overlap: int):
    """break text into overlapping chunks"""
    if size <= 0: 
        raise ValueError("chunk size must be > 0")
    
    text = text or ""
    n = len(text)
    start = 0
    
    while start < n:
        end = min(start + size, n)
        yield text[start:end]
        
        if end >= n:
            break
        
        start = max(0, end - overlap)


def cosine_sim(a, b):
    """calculate similarity scores"""
    import numpy as np
    
    a = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    
    return (b_norm @ a).tolist()


def within_registrable_domain(url: str, root_regdom: str) -> bool:
    """check if url is in same domain"""
    import tldextract, urllib.parse
    try:
        netloc = urllib.parse.urlparse(url).netloc
        ext = tldextract.extract(netloc)
        regdom = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
        return regdom == root_regdom
    except Exception:
        return False
