"""
Mental Health Data Platform — Ask Anything Agent
=================================================
A hosted Streamlit app that gives anyone access to the full CDC + SAMHSA
mental-health data catalog via natural-language questions. No local setup,
no data download, no PostgreSQL, no MinIO required.

How it works:
  1. On first load, the app discovers every mental-health dataset in the
     CDC + SAMHSA Socrata catalogs across ~30 topic queries (~500 datasets).
  2. It embeds each dataset's metadata (title + description + columns) with
     OpenAI and keeps the index in Streamlit's resource cache.
  3. When a user asks a question, the app semantic-matches it against the
     catalog, fetches fresh sample rows live from Socrata for the top-matching
     datasets, and answers with citations.

Deploy on Streamlit Community Cloud (free):
  - Main file: demo.py
  - Secret:    OPENAI_API_KEY = "sk-..."
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Data — Ask Anything",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Config ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
OPENAI_MODEL: str = "gpt-4o-mini"
EMBED_MODEL: str = "text-embedding-3-small"
EMBED_DIM: int = 1536

DOMAINS: Dict[str, str] = {
    "CDC": "data.cdc.gov",
    "SAMHSA": "data.samhsa.gov",
}

# Diverse mental-health topic seeds — used to sweep the Socrata catalogs so the
# resulting index covers "everything" (depression, anxiety, suicide, substance
# use, youth, veterans, opioids, access to care, disparities, etc.)
SEED_QUERIES: List[str] = [
    "mental health", "depression", "anxiety", "suicide", "self harm",
    "substance abuse", "substance use disorder", "opioid", "overdose",
    "alcohol use", "drug use", "tobacco cessation",
    "youth mental health", "adolescent depression", "child behavioral health",
    "adult mental illness", "serious mental illness", "psychological distress",
    "veterans mental health", "PTSD", "trauma",
    "mental health treatment", "behavioral health services", "psychiatric",
    "mental health workforce", "access to care", "insurance coverage mental health",
    "mental health disparities", "rural mental health",
    "loneliness", "wellbeing", "stress", "sleep",
    "eating disorder", "ADHD", "bipolar", "schizophrenia",
]

DATASETS_PER_QUERY_PER_ORG: int = 10
CATALOG_TTL_SECONDS: int = 60 * 60 * 6   # rebuild the catalog every 6 hours
SAMPLE_ROWS_FOR_ANSWER: int = 60         # how many rows to fetch per matched dataset
TOP_DATASETS_FOR_ANSWER: int = 3         # number of datasets to consult per question


# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0b1020,#141a3a); }
[data-testid="stSidebar"] * { color: #e7ecff !important; }
.big-hero {
    background: linear-gradient(135deg,#1a1f4a 0%,#2b1e5e 100%);
    border: 1px solid rgba(124,92,255,0.35);
    border-radius: 18px; padding: 28px 32px; margin-bottom: 24px;
}
.big-hero h1 { margin: 0 0 6px; font-size: 30px; }
.big-hero p  { margin: 0; color: #b8c1ee; }
.dataset-card {
    background: rgba(124,92,255,0.06);
    border: 1px solid rgba(124,92,255,0.20);
    border-radius: 12px; padding: 12px 14px; margin-bottom: 8px;
}
.dataset-card b { color: #d7ceff; }
.answer-box {
    background: rgba(37,208,164,0.06);
    border: 1px solid rgba(37,208,164,0.30);
    border-radius: 12px; padding: 18px 20px; margin: 10px 0 22px;
    font-size: 16px; line-height: 1.55;
}
.metric-pill {
    display: inline-block; padding: 4px 10px; border-radius: 999px;
    background: rgba(124,92,255,0.15); color: #c2b3ff;
    font-size: 12px; margin-right: 6px;
}
</style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Socrata catalog + data fetch
# ══════════════════════════════════════════════════════════════════════════════

def _search_one(query: str, org: str, limit: int) -> List[Dict[str, Any]]:
    domain = DOMAINS[org]
    try:
        resp = requests.get(
            "https://api.us.socrata.com/api/catalog/v1",
            params={"q": query, "domains": domain, "only": "datasets", "limit": limit},
            timeout=20,
        )
        resp.raise_for_status()
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for item in resp.json().get("results", []):
        r = item.get("resource", {}) or {}
        uid = r.get("id", "")
        if not uid:
            continue
        columns = r.get("columns_name", []) or []
        out.append({
            "uid": uid,
            "org": org,
            "name": (r.get("name") or "Untitled").strip(),
            "description": ((r.get("description") or "") or "").strip()[:800],
            "columns": columns[:40],
            "link": item.get("link") or f"https://{domain}/d/{uid}",
        })
    return out


@st.cache_data(ttl=CATALOG_TTL_SECONDS, show_spinner=False)
def discover_catalog() -> List[Dict[str, Any]]:
    """Sweep CDC + SAMHSA across all seed queries and return unique datasets."""
    seen: Dict[str, Dict[str, Any]] = {}
    tasks: List[Tuple[str, str]] = [(q, org) for q in SEED_QUERIES for org in DOMAINS]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_search_one, q, org, DATASETS_PER_QUERY_PER_ORG): (q, org)
            for q, org in tasks
        }
        for fut in as_completed(futures):
            for ds in fut.result():
                seen.setdefault(ds["uid"], ds)
    return list(seen.values())


def _metadata_text(ds: Dict[str, Any]) -> str:
    cols = ", ".join(ds.get("columns", [])) or "(columns unknown)"
    return (
        f"[{ds['org']}] {ds['name']}\n"
        f"Description: {ds['description'] or '(no description)'}\n"
        f"Columns: {cols}"
    )


def fetch_sample_rows(uid: str, org: str, limit: int = SAMPLE_ROWS_FOR_ANSWER) -> pd.DataFrame:
    domain = DOMAINS.get(org, "data.cdc.gov")
    try:
        resp = requests.get(
            f"https://{domain}/resource/{uid}.json",
            params={"$limit": limit},
            timeout=25,
        )
        resp.raise_for_status()
        return pd.DataFrame(resp.json())
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# OpenAI
# ══════════════════════════════════════════════════════════════════════════════

def _openai_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}


def embed_texts(texts: List[str], batch_size: int = 96) -> np.ndarray:
    """Batch-embed texts. Returns an (N, EMBED_DIM) numpy array."""
    if not OPENAI_API_KEY:
        return np.random.rand(len(texts), EMBED_DIM).astype(np.float32)
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=_openai_headers(),
            json={"model": EMBED_MODEL, "input": batch},
            timeout=60,
        )
        resp.raise_for_status()
        out.extend(d["embedding"] for d in resp.json()["data"])
    return np.asarray(out, dtype=np.float32)


def chat_completion(prompt: str, system: str) -> str:
    if not OPENAI_API_KEY:
        return "⚠️ OpenAI key not configured. Add `OPENAI_API_KEY` to Streamlit secrets."
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=_openai_headers(),
        json={
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 900,
        },
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ══════════════════════════════════════════════════════════════════════════════
# Catalog index — built once, cached across sessions
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def build_catalog_index() -> Dict[str, Any]:
    """Discover every mental-health dataset and embed its metadata."""
    datasets = discover_catalog()
    if not datasets:
        return {"datasets": [], "embeddings": np.zeros((0, EMBED_DIM), dtype=np.float32)}
    texts = [_metadata_text(d) for d in datasets]
    embeddings = embed_texts(texts)
    # L2-normalize once so ranking = a dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    return {"datasets": datasets, "embeddings": embeddings}


def rank_datasets(question: str, index: Dict[str, Any], k: int) -> List[Tuple[float, Dict[str, Any]]]:
    if not index["datasets"]:
        return []
    q_vec = embed_texts([question])[0]
    n = np.linalg.norm(q_vec)
    if n == 0:
        return []
    q_vec = q_vec / n
    scores = index["embeddings"] @ q_vec
    top_idx = np.argsort(-scores)[:k]
    return [(float(scores[i]), index["datasets"][i]) for i in top_idx]


# ══════════════════════════════════════════════════════════════════════════════
# Answering
# ══════════════════════════════════════════════════════════════════════════════

def _dataframe_to_context(df: pd.DataFrame, name: str, max_rows: int = 40) -> str:
    if df.empty:
        return f"[{name}] (no rows returned)"
    df = df.head(max_rows)
    lines = [f"[{name}]  columns: {', '.join(df.columns[:20])}"]
    for _, row in df.iterrows():
        cells = [f"{k}={v}" for k, v in row.items() if pd.notna(v) and str(v).strip()][:12]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def answer_question(question: str, index: Dict[str, Any]) -> Dict[str, Any]:
    ranked = rank_datasets(question, index, TOP_DATASETS_FOR_ANSWER)
    if not ranked:
        return {"answer": "No datasets available.", "sources": [], "contexts": []}

    contexts: List[str] = []
    sources: List[Dict[str, Any]] = []
    for score, ds in ranked:
        df = fetch_sample_rows(ds["uid"], ds["org"])
        contexts.append(_dataframe_to_context(df, ds["name"]))
        sources.append({"score": score, "dataset": ds, "rows": len(df)})

    system = (
        "You are an expert public-health data analyst. Answer the user's question "
        "using ONLY the provided dataset excerpts. Cite the dataset names in "
        "brackets. If the data does not directly answer the question, say so and "
        "explain what the data *does* show."
    )
    prompt = (
        f"Question: {question}\n\n"
        f"Data excerpts from CDC and SAMHSA:\n\n"
        + "\n\n---\n\n".join(contexts)
        + "\n\nAnswer:"
    )
    answer = chat_completion(prompt, system)
    return {"answer": answer, "sources": sources, "contexts": contexts}


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# Sidebar
with st.sidebar:
    st.markdown("## 🧠 Mental Health Data")
    st.caption("Ask any question about US mental-health statistics.")
    st.divider()

    if OPENAI_API_KEY:
        st.success("✅ OpenAI connected")
    else:
        st.error("❌ Add `OPENAI_API_KEY` to Streamlit secrets")

    st.markdown("### Data sources")
    st.markdown("- **CDC** — data.cdc.gov\n- **SAMHSA** — data.samhsa.gov")

    st.divider()
    if st.button("🔄 Rebuild catalog"):
        discover_catalog.clear()
        build_catalog_index.clear()
        st.rerun()

# Hero
st.markdown(
    """
<div class="big-hero">
  <h1>🧠 Ask anything about US mental health data</h1>
  <p>Powered by the full CDC and SAMHSA public data catalogs. Nothing to install — no data to download. Just ask.</p>
</div>
    """,
    unsafe_allow_html=True,
)

# Build (or load cached) catalog
with st.spinner("Loading the CDC + SAMHSA mental-health catalog… (first load: ~1 min, then cached)"):
    t0 = time.time()
    index = build_catalog_index()
    build_ms = (time.time() - t0) * 1000

n_datasets = len(index["datasets"])
if n_datasets == 0:
    st.error("Could not load the catalog. Try the **Rebuild catalog** button in the sidebar.")
    st.stop()

st.markdown(
    f'<span class="metric-pill">📚 {n_datasets} datasets indexed</span>'
    f'<span class="metric-pill">⚡ loaded in {build_ms/1000:.1f}s</span>'
    f'<span class="metric-pill">🏛️ CDC + SAMHSA</span>',
    unsafe_allow_html=True,
)

# Example prompts
st.write("")
example_cols = st.columns(4)
examples = [
    "What are the trends in youth depression?",
    "Which states have the highest opioid overdose rates?",
    "How has veteran suicide changed over time?",
    "What's the gap in mental-health treatment access by race?",
]
for col, ex in zip(example_cols, examples):
    with col:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex

# Main input
question = st.text_input(
    "Your question",
    key="question",
    placeholder="e.g. What percentage of adolescents received mental-health services last year?",
)

if st.button("💬 Answer", type="primary") and question.strip():
    with st.spinner("Finding the most relevant datasets and generating an answer…"):
        result = answer_question(question, index)

    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

    st.markdown("### 📚 Sources consulted")
    for src in result["sources"]:
        ds = src["dataset"]
        st.markdown(
            f"""<div class="dataset-card">
<b>{ds['name']}</b> <span class="metric-pill">{ds['org']}</span>
<span class="metric-pill">match {src['score']:.2f}</span>
<span class="metric-pill">{src['rows']} rows sampled</span>
<br><small>{(ds['description'] or '(no description)')[:280]}</small>
<br><a href="{ds['link']}" target="_blank">Open on {ds['org']} ↗</a>
</div>""",
            unsafe_allow_html=True,
        )

    with st.expander("🔍 Raw data excerpts used"):
        for ctx in result["contexts"]:
            st.text(ctx[:1500])
            st.divider()

# Catalog browser (collapsed by default)
with st.expander(f"📖 Browse the full catalog of {n_datasets} datasets"):
    df = pd.DataFrame([
        {"Org": d["org"], "Name": d["name"], "UID": d["uid"], "Link": d["link"]}
        for d in index["datasets"]
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
