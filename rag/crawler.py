import json, os, time, urllib.parse, urllib.robotparser, requests
from bs4 import BeautifulSoup
from readability import Document
from typing import Dict, List, Set, Tuple
from .utils import normalize_ws, canonical_url, within_registrable_domain

# use browser headers so sites don't block us
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def can_fetch(url: str, user_agent: str) -> bool:
    """check robots.txt before crawling"""
    parsed = urllib.parse.urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True  # if can't read robots.txt, proceed anyway


def extract_main_text(html: str):
    """extract main content, remove nav/ads/etc"""
    try:
        # readability library works pretty well
        doc = Document(html)
        title = doc.short_title() or ""
        content_html = doc.summary()
        soup = BeautifulSoup(content_html, "lxml")
        for tag in soup(["script","style","noscript","header","footer","nav","aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        return title, normalize_ws(text)
    except Exception:
        # fallback method
        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.string if soup.title else "") or ""
        for tag in soup(["script","style","noscript","header","footer","nav","aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        return title, normalize_ws(text)

def crawl(start_url: str, max_pages: int = 30, max_depth: int = 3, crawl_delay_ms: int = 500) -> Dict:
    """breadth-first crawl staying in same domain"""
    import tldextract
    
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    start = canonical_url(start_url)
    parsed = urllib.parse.urlparse(start)
    
    if not parsed.scheme.startswith("http"):
        raise ValueError("start_url must be http(s)")
    
    # get the root domain
    ext = tldextract.extract(parsed.netloc)
    root_regdom = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    
    queue: List[tuple] = [(start, 0)]
    seen: Set[str] = set()
    pages: List[Dict] = []
    skipped = 0
    
    while queue and len(pages) < max_pages:
        url, depth = queue.pop(0)
        url = canonical_url(url)
        
        if url in seen: 
            continue
        seen.add(url)
        
        if depth > max_depth: 
            continue
        if not within_registrable_domain(url, root_regdom): 
            continue
        if not can_fetch(url, DEFAULT_HEADERS["User-Agent"]):
            skipped += 1
            continue
        
        try:
            resp = session.get(url, timeout=15)
            
            if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type",""):
                skipped += 1
                continue
            
            title, text = extract_main_text(resp.text)
            
            if len(text) < 80:  # skip pages with barely any content
                skipped += 1
                continue
            
            pages.append({"url": url, "title": title, "text": text})
            
            # find links on this page
            soup = BeautifulSoup(resp.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = urllib.parse.urljoin(url, a["href"])
                if href.startswith(("mailto:","javascript:")): 
                    continue
                if within_registrable_domain(href, root_regdom):
                    queue.append((href, depth+1))
            
            time.sleep(crawl_delay_ms / 1000.0)  # be polite
            
        except Exception:
            skipped += 1
            continue
    
    # save to file
    os.makedirs("data", exist_ok=True)
    with open("data/pages.jsonl","w",encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False)+"\n")
    
    return {"page_count": len(pages), "skipped_count": skipped, "urls": [p["url"] for p in pages]}

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_url", required=True)
    ap.add_argument("--max_pages", type=int, default=30)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--crawl_delay_ms", type=int, default=500)
    args = ap.parse_args()
    print(json.dumps(crawl(args.start_url, args.max_pages, args.max_depth, args.crawl_delay_ms), indent=2))
