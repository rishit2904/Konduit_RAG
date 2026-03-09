import os, requests
from typing import Dict, List
from .prompt import SYSTEM_PROMPT, USER_TEMPLATE


def _openai_chat(messages, model, api_key, api_base=None):
    url = (api_base or "https://api.openai.com/v1") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.0}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]


def _ollama_chat(messages, model):
    """try local ollama if available"""
    url = "http://localhost:11434/api/generate"
    prompt = ""
    for m in messages:
        prompt += f"[{m['role'].upper()}]\\n{m['content']}\\n\\n"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response","")


def generate_answer(question: str, retrieved: List[Dict]) -> str:
    """generate answer from chunks - tries openai, then ollama, then just returns snippets"""
    context = "\n".join([f"[{r['url']}] {r['snippet']}" for r in retrieved[:10]])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(question=question, context=context)},
    ]
    
    # try openai first
    api_key = os.getenv("OPENAI_API_KEY","").strip()
    model = os.getenv("OPENAI_MODEL","gpt-4o-mini").strip()
    api_base = os.getenv("OPENAI_API_BASE","").strip()
    
    if api_key:
        try:
            return _openai_chat(messages, model, api_key, api_base or None)
        except Exception:
            pass
    
    # try ollama
    ollama_model = os.getenv("OLLAMA_MODEL","").strip()
    if ollama_model:
        try:
            return _ollama_chat(messages, ollama_model)
        except Exception:
            pass
    
    # fallback: just return snippets
    if not context.strip():
        return "not found in crawled content"
    
    synthesis = " ".join([r["snippet"] for r in retrieved[:2]])
    return synthesis[:800]
