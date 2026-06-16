"""
Mental Health Data Platform — Standalone Demo
Runs entirely from Streamlit Cloud: queries Socrata + OpenAI directly.
No backend, no PostgreSQL, no MinIO required.
Set OPENAI_API_KEY as a Streamlit secret to enable AI answers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from typing import List, Dict, Any, Optional

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 Mental Health Data Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── OpenAI key (from Streamlit secrets or env var) ─────────────────────────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
OPENAI_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# ── Socrata catalog domains ─────────────────────────────────────────────────────
DOMAINS = {
    "CDC": "data.cdc.gov",
    "SAMHSA": "data.samhsa.gov",
}

# ── CSS ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f0f1e,#1a1a2e); }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.metric-card {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(102,126,234,0.3);
    margin-bottom: 8px;
}
.result-card {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 14px;
    border-left: 3px solid #667eea;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Socrata helpers
# ══════════════════════════════════════════════════════════════════════════════

def search_socrata_catalog(query: str, org: str, limit: int = 15) -> List[Dict]:
    """Search the Socrata unified catalog for datasets."""
    domain = DOMAINS.get(org, "data.cdc.gov")
    try:
        resp = requests.get(
            "https://api.us.socrata.com/api/catalog/v1",
            params={"q": query, "domains": domain, "only": "datasets", "limit": limit},
            timeout=20,
        )
        resp.raise_for_status()
        results = []
        for item in resp.json().get("results", []):
            r = item.get("resource", {})
            uid = r.get("id", "")
            if uid:
                results.append({
                    "uid": uid,
                    "name": r.get("name", "Unknown"),
                    "description": (r.get("description") or "")[:300],
                    "org": org,
                    "link": item.get("link") or f"https://{domain}/d/{uid}",
                    "rows": r.get("page_views", {}).get("page_views_total", "?"),
                })
        return results
    except Exception as e:
        st.error(f"Catalog search failed: {e}")
        return []


def fetch_dataset_preview(uid: str, org: str, limit: int = 200) -> pd.DataFrame:
    """Fetch rows from Socrata using the SODA API."""
    domain = DOMAINS.get(org, "data.cdc.gov")
    try:
        resp = requests.get(
            f"https://{domain}/resource/{uid}.json",
            params={"$limit": limit},
            timeout=30,
        )
        resp.raise_for_status()
        return pd.DataFrame(resp.json())
    except Exception as e:
        st.warning(f"Could not fetch dataset: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# OpenAI helpers
# ══════════════════════════════════════════════════════════════════════════════

def _openai_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Create embeddings via OpenAI API."""
    if not OPENAI_API_KEY:
        # Random fallback for demo when no key
        return [np.random.rand(EMBED_DIM).tolist() for _ in texts]
    resp = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers=_openai_headers(),
        json={"model": EMBED_MODEL, "input": texts},
        timeout=30,
    )
    resp.raise_for_status()
    return [d["embedding"] for d in resp.json()["data"]]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def chat_completion(prompt: str, system: str = "You are a helpful public health data analyst.") -> str:
    """Single-turn chat via OpenAI."""
    if not OPENAI_API_KEY:
        return "⚠️ No OpenAI API key configured. Add `OPENAI_API_KEY` to your Streamlit secrets to enable AI answers."
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=_openai_headers(),
        json={
            "model": OPENAI_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 800,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ══════════════════════════════════════════════════════════════════════════════
# In-session RAG
# ══════════════════════════════════════════════════════════════════════════════

def build_chunks_from_df(df: pd.DataFrame, dataset_name: str, chunk_size: int = 30) -> List[Dict]:
    """Convert a DataFrame into text chunks for RAG."""
    chunks = []
    cols = list(df.columns)
    for i in range(0, min(len(df), 300), chunk_size):
        sub = df.iloc[i : i + chunk_size]
        lines = [f"Dataset: {dataset_name}", f"Rows {i+1}–{min(i+chunk_size, len(df))}:", ""]
        for _, row in sub.iterrows():
            row_str = " | ".join(
                f"{k}: {v}" for k, v in row.items()
                if pd.notna(v) and str(v).strip()
            )
            if row_str:
                lines.append(row_str)
        chunks.append({"text": "\n".join(lines), "source": dataset_name, "rows": (i, i + chunk_size)})
    return chunks


def rag_index(chunks: List[Dict]) -> List[Dict]:
    """Embed all chunks and return them with embeddings."""
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    for c, emb in zip(chunks, embeddings):
        c["embedding"] = emb
    return chunks


def rag_search(query: str, indexed_chunks: List[Dict], k: int = 5) -> List[Dict]:
    """Find the k most relevant chunks for query."""
    [q_emb] = embed_texts([query])
    scored = [(cosine_similarity(q_emb, c["embedding"]), c) for c in indexed_chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


def rag_answer(question: str, top_chunks: List[Dict]) -> str:
    context = "\n\n".join(f"[Source {i+1}]\n{c['text'][:600]}" for i, c in enumerate(top_chunks))
    prompt = f"""You are an expert mental health data analyst.
Answer the question using ONLY the provided context data.
Be specific — cite numbers, trends, and locations from the data.
If the answer isn't in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
    return chat_completion(prompt)


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "search_results": [],
        "selected_dataset": None,
        "preview_df": pd.DataFrame(),
        "rag_index": [],
        "rag_indexed_name": "",
        "chat_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 Mental Health Platform")
    st.markdown("**RAG-powered exploration of CDC & SAMHSA mental health datasets**")
    st.divider()

    if OPENAI_API_KEY:
        st.success("✅ OpenAI connected")
    else:
        st.warning("⚠️ No OpenAI key — add `OPENAI_API_KEY` to Streamlit secrets for AI features")

    st.divider()
    st.markdown("### Data Sources")
    st.markdown("- 🏛️ **CDC** — data.cdc.gov")
    st.markdown("- 🏥 **SAMHSA** — data.samhsa.gov")
    st.divider()
    st.markdown("### How to use")
    st.markdown("1. **Search** for datasets\n2. **Preview** a dataset\n3. **Index** it for RAG\n4. **Ask questions** about the data")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

st.title("🧠 Mental Health Data Platform")
st.caption("Explore CDC & SAMHSA datasets with AI-powered question answering")

tab_search, tab_preview, tab_ask = st.tabs(["🔍 Search Datasets", "📊 Preview & Index", "💬 Ask Questions"])


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Search
# ──────────────────────────────────────────────────────────────────────────────
with tab_search:
    st.subheader("Search Mental Health Datasets")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input("Search query", placeholder="e.g. youth depression, substance abuse, suicide rates")
    with col2:
        org = st.selectbox("Source", list(DOMAINS.keys()))
    with col3:
        limit = st.slider("Max results", 5, 30, 12)

    if st.button("🔍 Search", type="primary"):
        with st.spinner(f"Searching {org} catalog..."):
            st.session_state.search_results = search_socrata_catalog(query, org, limit)

    results = st.session_state.search_results
    if results:
        st.markdown(f"**{len(results)} datasets found**")
        for i, ds in enumerate(results):
            with st.container():
                st.markdown(f"""<div class="result-card">
<b>{ds['name']}</b> &nbsp;|&nbsp; <code>{ds['uid']}</code> &nbsp;|&nbsp; {ds['org']}
<br><small>{ds['description'] or 'No description'}</small>
</div>""", unsafe_allow_html=True)
                col_a, col_b = st.columns([1, 4])
                with col_a:
                    if st.button("Load this dataset →", key=f"load_{i}"):
                        st.session_state.selected_dataset = ds
                        st.session_state.preview_df = pd.DataFrame()
                        st.session_state.rag_index = []
                        st.success(f"Selected: **{ds['name']}** — go to the Preview & Index tab")
                with col_b:
                    st.markdown(f"[Open on {ds['org']} ↗]({ds['link']})")
    elif not query:
        st.info("Enter a query above to search the dataset catalog.")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — Preview & Index
# ──────────────────────────────────────────────────────────────────────────────
with tab_preview:
    ds = st.session_state.selected_dataset

    if not ds:
        st.info("Select a dataset from the Search tab first.")
    else:
        st.subheader(f"📊 {ds['name']}")
        st.caption(f"Source: {ds['org']} | ID: `{ds['uid']}`")
        col_a, col_b = st.columns(2)
        with col_a:
            n_rows = st.slider("Rows to fetch", 50, 1000, 200, step=50)
        with col_b:
            st.markdown(f"[View on {ds['org']} ↗]({ds['link']})")

        if st.button("⬇️ Load / Refresh Preview", type="primary"):
            with st.spinner("Fetching data from Socrata..."):
                df = fetch_dataset_preview(ds["uid"], ds["org"], n_rows)
                st.session_state.preview_df = df
                st.session_state.rag_index = []

        df = st.session_state.preview_df
        if not df.empty:
            st.markdown(f"**{len(df)} rows × {len(df.columns)} columns**")
            st.dataframe(df.head(50), use_container_width=True)

            # Quick visualisation
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if num_cols:
                with st.expander("📈 Quick chart"):
                    import plotly.express as px
                    col_to_plot = st.selectbox("Column", num_cols, key="chart_col")
                    fig = px.histogram(df, x=col_to_plot, title=f"Distribution of {col_to_plot}")
                    st.plotly_chart(fig, use_container_width=True)

            # Index for RAG
            st.divider()
            st.markdown("### 🤖 Index for AI Q&A")
            st.caption("Splits the dataset into text chunks and creates semantic embeddings so you can ask natural-language questions.")
            if st.button("⚡ Index this dataset for RAG", type="secondary"):
                with st.spinner("Chunking + embedding (may take 30–60 s)..."):
                    try:
                        chunks = build_chunks_from_df(df, ds["name"])
                        indexed = rag_index(chunks)
                        st.session_state.rag_index = indexed
                        st.session_state.rag_indexed_name = ds["name"]
                        st.success(f"✅ Indexed {len(indexed)} chunks from **{ds['name']}** — go to Ask Questions tab!")
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")

            if st.session_state.rag_index and st.session_state.rag_indexed_name == ds["name"]:
                st.success(f"✅ {len(st.session_state.rag_index)} chunks indexed and ready")
        else:
            st.info("Click 'Load / Refresh Preview' to fetch data.")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — Ask Questions
# ──────────────────────────────────────────────────────────────────────────────
with tab_ask:
    st.subheader("💬 Ask Questions About Your Data")

    idx = st.session_state.rag_index
    name = st.session_state.rag_indexed_name

    if not idx:
        st.info("Load and index a dataset from the **Preview & Index** tab first.")
    else:
        st.success(f"🗄️ Active dataset: **{name}** ({len(idx)} chunks indexed)")
        st.divider()

        question = st.text_input("Your question", placeholder="e.g. What are the top states for depression? What trends exist across years?")
        k = st.slider("Chunks to retrieve", 3, 10, 5)

        if st.button("💬 Get AI Answer", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching relevant data + generating answer..."):
                    try:
                        top_chunks = rag_search(question, idx, k)
                        answer = rag_answer(question, top_chunks)
                        st.session_state.chat_history.append({"q": question, "a": answer, "sources": top_chunks})
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Display chat history newest-first
        for entry in reversed(st.session_state.chat_history):
            st.markdown(f"**Q:** {entry['q']}")
            st.markdown(f"**A:** {entry['a']}")
            with st.expander("📄 Retrieved data chunks"):
                for i, c in enumerate(entry["sources"]):
                    st.text(f"[Chunk {i+1} — rows {c.get('rows','?')}]\n{c['text'][:400]}")
            st.divider()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear history"):
                st.session_state.chat_history = []
                st.rerun()
