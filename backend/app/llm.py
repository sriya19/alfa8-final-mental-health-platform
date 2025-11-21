from typing import List, Dict, Any
from .config import settings

# ---------- Persona-aware system prompt ----------
PERSONA_SYSTEM_PROMPTS = {
    "policy maker": """You are a concise public-policy briefing assistant.
Speak in plain language, avoid jargon, and emphasize implications, trends, and actions.
Prefer bullets, short paragraphs, and a 1-2 sentence bottom line.""",

    "clinician": """You are a clinician-facing assistant.
Be medically precise, cite data limitations, risk factors, and patient-impact.
Prioritize actionable guidance, red flags, and clinical caveats in neutral tone.""",

    "researcher": """You are a research assistant.
Be rigorous and cautious, highlight methodology, data quality, uncertainty, and confounders.
Use technical language when appropriate and surface gaps & next-steps for study.""",
}

def build_persona_system_prompt(persona: str | None) -> str:
    if not persona:
        return ("You are a helpful assistant for public-health analytics. "
                "Be clear, cautious, and note data limitations.")
    key = (persona or "").strip().lower()
    return PERSONA_SYSTEM_PROMPTS.get(
        key,
        "You are a helpful assistant for public-health analytics. Be clear and note limitations."
    )

# ---------- RAG user prompt template ----------
def build_rag_user_prompt(question: str, snippets: list[dict]) -> str:
    """
    snippets: [{ 'uid': 'xkb8-kh2a', 'name': '...', 'description': '...', 'link': '...' }, ...]
    Only pass a handful (k) of the most relevant items.
    """
    lines = []
    lines.append("You are given context snippets from CDC/SAMHSA datasets.")
    lines.append("Use them to answer the user’s question faithfully; if unsure, say so.")
    lines.append("")
    lines.append("=== Context Snippets ===")
    for i, s in enumerate(snippets, 1):
        uid = s.get("uid", "")
        name = s.get("name", "")
        desc = (s.get("description", "") or "").strip()
        link = s.get("link", "")
        # keep context tight
        desc = desc[:1200]
        lines.append(f"[{i}] {name} (uid: {uid})")
        if link:
            lines.append(f"    {link}")
        if desc:
            lines.append(f"    {desc}")
        lines.append("")
    lines.append("=== Task ===")
    lines.append(
        "Answer the question using ONLY the information that can be reasonably inferred from the snippets. "
        "Call out important limitations or caveats. If data are provisional, say so.")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("In your answer:")
    lines.append("- Start with a 1-sentence bottom line tailored to the persona.")
    lines.append("- Then provide 3–6 bullet points grounded in the snippets.")
    lines.append("- End with a brief 'Limitations & Next steps' section.")
    return "\n".join(lines)

try:
    from openai import OpenAI
except Exception:  # package not installed yet
    OpenAI = None  # type: ignore

def have_llm() -> bool:
    return bool(settings.OPENAI_API_KEY) and OpenAI is not None

def get_client():
    if not have_llm():
        raise RuntimeError("LLM not configured: missing OPENAI_API_KEY or package")
    return OpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or "https://api.openai.com/v1"
    )

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Returns list of embedding vectors (length = settings.EMBEDDING_DIM).
    """
    client = get_client()
    model = settings.EMBEDDING_MODEL or "text-embedding-3-small"
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def generate_answer(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Simple RAG answer composer: sends the top chunks as context.
    """
    client = get_client()
    sys_prompt = (
        "You are a careful public health analyst. "
        "Answer the user's question using ONLY the provided context. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Cite sources by chunk_id like [chunk:<id>]."
    )
    context = ""
    for c in context_chunks:
        cid = c.get("chunk_id")
        txt = c.get("content", "")[:3000]
        context += f"\n[chunk:{cid}] {txt}\n"

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
    ]
    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL or "gpt-4o-mini",
        messages=messages,
        temperature=0.2
    )
    return resp.choices[0].message.content or ""

# --- Chat completion helper (async, via httpx) ---

import os, time, httpx
from tenacity import retry, stop_after_attempt, wait_exponential

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _chat_price(model: str) -> dict:
    # fallback prices if not already defined above in this file
    # (you already had a CHAT_PRICES_PER_1K dict; this just guards in case)
    default = {"input": 0.00015, "output": 0.00060}  # gpt-4o-mini
    try:
        return CHAT_PRICES_PER_1K.get(model, default)  # type: ignore[name-defined]
    except Exception:
        return default

def _usage_dict(resp_json: dict) -> dict:
    # compatible with OpenAI responses; you already had a version—this is a safe fallback
    return resp_json.get("usage", {}) if isinstance(resp_json, dict) else {}

def _should_retry_httpx(exc: Exception) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (429, 500, 502, 503, 504)

@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=1, max=60), reraise=True)
async def chat_completion(
    prompt: str,
    *,
    system: str | None = "You are a helpful assistant for public health analytics.",
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = 500,
) -> str:
    """
    Minimal async chat completion using OpenAI's /chat/completions.
    Returns the assistant message content string.
    Also prints a cost log based on token usage and simple pricing table.
    """
    model = model or OPENAI_MODEL
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    t0 = time.time()
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            if _should_retry_httpx(e):
                raise
            raise
        data = r.json()

    # extract text
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()

    # cost logging
    usage = _usage_dict(data)
    in_toks = usage.get("prompt_tokens", 0)
    out_toks = usage.get("completion_tokens", 0)
    prices = _chat_price(model)
    est_cost = (in_toks / 1000.0) * prices["input"] + (out_toks / 1000.0) * prices["output"]
    dt = time.time() - t0
    print(f"[COST LOG] Chat model={model} tokens in/out: {in_toks}/{out_toks} | ${est_cost:.6f} | {dt:.2f}s")

    return content
