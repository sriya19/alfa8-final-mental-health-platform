"""
Mental Health Data Platform — Full-Featured Standalone Demo
All 6 tabs from the original app, self-contained (no separate backend needed).
Runs on Streamlit Cloud with just OPENAI_API_KEY in secrets.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests, json, os, io, time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 Mental Health Data Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Secrets / env vars ─────────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
OPENAI_MODEL   = "gpt-4o-mini"
EMBED_MODEL    = "text-embedding-3-small"
EMBED_DIM      = 1536

DOMAINS = {"CDC": "data.cdc.gov", "SAMHSA": "data.samhsa.gov", "Maryland": "data.maryland.gov"}

PERSONAS = [
    "Public health researcher",
    "Policy maker",
    "Clinician",
    "Epidemiologist",
    "Data analyst",
]

PERSONA_DESC = {
    "Public health researcher": "📊 Statistical analysis, research methodology, evidence-based insights",
    "Policy maker":            "📋 Actionable recommendations, funding priorities, population impact",
    "Clinician":               "🏥 Clinical implications, treatment guidelines, patient care insights",
    "Epidemiologist":          "📈 Disease patterns, risk factors, population health metrics",
    "Data analyst":            "💻 Technical insights, data quality, trends and correlations",
}

# ── CSS ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"]   { background: linear-gradient(180deg,#0f0f1e,#1a1a2e); }
[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
.result-card  { background:rgba(255,255,255,.04); border-radius:10px; padding:14px;
                border-left:3px solid #667eea; margin-bottom:10px; }
.data-card    { background:rgba(102,126,234,.12); border-radius:12px; padding:16px;
                border:1px solid rgba(102,126,234,.3); margin-bottom:12px; }
.metric-row   { display:flex; gap:12px; flex-wrap:wrap; }
</style>
""", unsafe_allow_html=True)

_key_ctr = {"n": 0}
def _k(p="k"):
    _key_ctr["n"] += 1
    return f"{p}_{_key_ctr['n']}"

# ══════════════════════════════════════════════════════════════════════════════
# ── Socrata helpers ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def socrata_search(query: str, org: str, limit: int = 15) -> List[Dict]:
    domain = DOMAINS.get(org, "data.cdc.gov")
    try:
        r = requests.get(
            "https://api.us.socrata.com/api/catalog/v1",
            params={"q": query, "domains": domain, "only": "datasets", "limit": limit},
            timeout=20,
        )
        r.raise_for_status()
        out = []
        for item in r.json().get("results", []):
            res = item.get("resource", {})
            uid = res.get("id", "")
            if uid:
                out.append({
                    "uid": uid, "name": res.get("name", "Unknown"),
                    "description": (res.get("description") or "")[:350],
                    "org": org,
                    "link": item.get("link") or f"https://{domain}/d/{uid}",
                })
        return out
    except Exception as e:
        st.error(f"Catalog search error: {e}")
        return []


def socrata_fetch(uid: str, org: str, limit: int = 500) -> pd.DataFrame:
    domain = DOMAINS.get(org, "data.cdc.gov")
    try:
        r = requests.get(
            f"https://{domain}/resource/{uid}.json",
            params={"$limit": limit},
            timeout=30,
        )
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception as e:
        st.warning(f"Could not fetch dataset: {e}")
        return pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# ── OpenAI helpers ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _oai_headers():
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not OPENAI_API_KEY:
        return [np.random.rand(EMBED_DIM).tolist() for _ in texts]
    r = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers=_oai_headers(),
        json={"model": EMBED_MODEL, "input": texts},
        timeout=40,
    )
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]


def chat(prompt: str, system: str = "You are a helpful public health data analyst.", max_tokens: int = 900) -> str:
    if not OPENAI_API_KEY:
        return "⚠️ Add `OPENAI_API_KEY` to Streamlit secrets to enable AI features."
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=_oai_headers(),
        json={
            "model": OPENAI_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "temperature": 0.3, "max_tokens": max_tokens,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def optimize_keywords(story: str, persona: str) -> List[str]:
    """Use AI to rewrite the user's story into good keyword search queries."""
    prompt = (
        f"You are helping a '{persona}' find mental-health datasets.\n"
        f"Rewrite this into 3–5 short keyword search queries (one per line, no bullets):\n{story}"
    )
    try:
        raw = chat(prompt, max_tokens=200)
        return [ln.strip() for ln in raw.strip().splitlines() if ln.strip()][:5]
    except Exception:
        return [story]


def ai_insights(results: List[Dict], persona: str, story: str) -> str:
    snippet = json.dumps(results[:4], indent=2)[:2000]
    prompt = (
        f"Persona: {persona}\nSearch: {story}\n\nDatasets found:\n{snippet}\n\n"
        "Give 3 bullet-point insights for this persona about what these datasets reveal "
        "and which to prioritise. Be concise."
    )
    return chat(prompt, max_tokens=350)


def analyze_dataset_with_ai(df: pd.DataFrame, name: str, persona: str) -> Dict:
    """Ask AI to summarise the dataset and recommend charts."""
    cols_info = []
    for col in df.columns[:25]:
        s = df[col]
        try: nuniq = int(s.nunique())
        except Exception: nuniq = None
        cols_info.append({"name": col, "dtype": str(s.dtype), "nunique": nuniq,
                          "sample": [str(v) for v in s.dropna().head(3).tolist()]})

    prompt = f"""You are a data visualisation expert. A {persona} is exploring this dataset:
Name: {name}
Shape: {df.shape[0]} rows × {df.shape[1]} columns
Columns: {json.dumps(cols_info, indent=2)[:3000]}

Return ONLY valid JSON with these keys:
{{
  "summary": "2-3 sentence plain-language description of what this data contains",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "visualizations": [
    {{
      "type": "bar|line|scatter|histogram|pie|heatmap",
      "title": "Chart title",
      "x": "column_name_or_null",
      "y": "column_name_or_null",
      "color": "column_name_or_null",
      "description": "what this chart shows"
    }}
  ]
}}
Recommend 3-5 charts that would be meaningful for a {persona}.
Only reference columns that exist in the list above."""
    try:
        raw = chat(prompt, max_tokens=800)
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = raw[:-3]
        return json.loads(raw.strip())
    except Exception as e:
        return {"error": str(e), "summary": "", "key_findings": [], "visualizations": []}


def _resolve_col(name: Optional[str], df: pd.DataFrame) -> Optional[str]:
    if not name or name == "null": return None
    if name in df.columns: return name
    low = name.lower()
    for c in df.columns:
        if c.lower() == low: return c
    return None


def make_chart(rec: Dict, df: pd.DataFrame) -> Optional[go.Figure]:
    ctype   = rec.get("type", "bar")
    title   = rec.get("title", "Chart")
    x_col   = _resolve_col(rec.get("x"), df)
    y_col   = _resolve_col(rec.get("y"), df)
    clr_col = _resolve_col(rec.get("color"), df)

    # Coerce numeric
    for c in [x_col, y_col]:
        if c and c in df.columns:
            try: df[c] = pd.to_numeric(df[c], errors="ignore")
            except Exception: pass

    try:
        kw = dict(title=title, template="plotly_dark", height=400)
        if ctype == "bar" and x_col and y_col:
            fig = px.bar(df, x=x_col, y=y_col, color=clr_col, **kw)
        elif ctype == "line" and x_col and y_col:
            fig = px.line(df, x=x_col, y=y_col, color=clr_col, **kw)
        elif ctype == "scatter" and x_col and y_col:
            fig = px.scatter(df, x=x_col, y=y_col, color=clr_col, **kw)
        elif ctype == "histogram" and x_col:
            fig = px.histogram(df, x=x_col, color=clr_col, **kw)
        elif ctype == "pie" and x_col and y_col:
            counts = df.groupby(x_col)[y_col].sum().reset_index()
            fig = px.pie(counts, names=x_col, values=y_col, **kw)
        elif ctype == "heatmap":
            num_df = df.select_dtypes(include="number")
            if len(num_df.columns) >= 2:
                fig = px.imshow(num_df.corr(), text_auto=True, aspect="auto", **kw)
            else:
                return None
        else:
            # Fallback: histogram of first numeric col
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if num_cols:
                fig = px.histogram(df, x=num_cols[0], **kw)
            else:
                return None
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        return fig
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# ── In-session RAG ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def df_to_chunks(df: pd.DataFrame, name: str, uid: str, chunk_size: int = 40) -> List[Dict]:
    chunks = []
    for i in range(0, min(len(df), 400), chunk_size):
        sub = df.iloc[i: i + chunk_size]
        lines = [f"Dataset: {name}", f"Rows {i+1}–{min(i+chunk_size, len(df))}:", ""]
        for _, row in sub.iterrows():
            r = " | ".join(f"{k}: {v}" for k, v in row.items() if pd.notna(v) and str(v).strip())
            if r: lines.append(r)
        chunks.append({"text": "\n".join(lines), "uid": uid, "name": name,
                       "rows": (i, min(i + chunk_size, len(df)))})
    return chunks


def rag_embed(chunks: List[Dict]) -> List[Dict]:
    texts = [c["text"] for c in chunks]
    embs  = embed_texts(texts)
    for c, e in zip(chunks, embs):
        c["emb"] = e
    return chunks


def cosine(a, b):
    a, b = np.array(a), np.array(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def rag_retrieve(question: str, index: List[Dict], k: int = 6) -> List[Dict]:
    [qv] = embed_texts([question])
    scored = sorted(index, key=lambda c: cosine(qv, c["emb"]), reverse=True)
    return scored[:k]


def rag_answer(question: str, chunks: List[Dict], persona: str, style: str = "Detailed") -> str:
    ctx = "\n\n".join(f"[Source {i+1} — {c['name']}]\n{c['text'][:600]}"
                      for i, c in enumerate(chunks))
    style_map = {
        "Concise":          "Keep your answer to 3–5 sentences.",
        "Technical":        "Use technical language, include statistics and confidence caveats.",
        "Executive Summary":"Write a 4-bullet executive summary with a one-line bottom line.",
        "Detailed":         "Provide a thorough, structured answer with sub-sections.",
    }
    system = (f"You are a public health expert answering as a {persona}. "
              f"{style_map.get(style, '')} "
              "Answer ONLY from the provided context. Cite data where possible.")
    prompt = f"Question: {question}\n\nContext:\n{ctx}"
    return chat(prompt, system=system, max_tokens=900)

# ══════════════════════════════════════════════════════════════════════════════
# ── Session state ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _init():
    defaults = {
        "search_results":  [],
        "ingested":        {},   # uid → {"df": df, "name": str, "org": str, "at": datetime}
        "rag_index":       [],   # flat list of embedded chunks across all ingested datasets
        "viz_datasets":    {},   # uid → {"df": df, "name": str, "analysis": dict}
        "qa_history":      [],
        "ai_cache":        {},   # uid → analysis dict
        "search_history":  [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ══════════════════════════════════════════════════════════════════════════════
# ── Sidebar ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 Mental Health Platform")
    st.divider()

    org = st.selectbox("🏢 Filter by Organisation",
                       ["All", "CDC", "SAMHSA", "Maryland"], key="org_sel")

    st.markdown("### 👤 Select Your Role")
    persona = st.selectbox("Persona:", PERSONAS, key="persona_sel")
    st.info(PERSONA_DESC.get(persona, ""))

    st.divider()
    st.markdown("### 📊 Session Stats")
    n_ing = len(st.session_state.ingested)
    n_chk = len(st.session_state.rag_index)
    c1, c2 = st.columns(2)
    c1.metric("Datasets", n_ing)
    c2.metric("RAG Chunks", n_chk)

    if not OPENAI_API_KEY:
        st.warning("⚠️ Add `OPENAI_API_KEY` to Streamlit secrets for AI features.")
    else:
        st.success("✅ OpenAI connected")

    st.divider()
    st.caption("All data is stored in your browser session. Refresh = fresh start.")

# ══════════════════════════════════════════════════════════════════════════════
# ── Header ─────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='background:linear-gradient(135deg,#667eea,#764ba2);padding:28px;
border-radius:15px;margin-bottom:24px;box-shadow:0 10px 30px rgba(0,0,0,.3)'>
<h1 style='color:white;text-align:center;margin:0'>🧠 Maryland Mental Health Data Platform</h1>
<p style='color:white;text-align:center;margin:8px 0 0 0;opacity:.9'>
AI-Powered Analysis • RAG-Enhanced Insights • CDC & SAMHSA Data</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "🔍 Search & Ingest",
    "📊 AI Visualizations",
    "💬 Ask Questions (AI)",
    "📈 Analytics Dashboard",
    "💾 Export & Share",
    "⚙️ Manage Data",
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 0 — Search & Ingest
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Find and Ingest Relevant Datasets")

    with st.form("search_form"):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"**Searching as:** {persona}")
        with col2:
            search_type = st.selectbox("Search Type",
                                       ["Keyword", "Semantic (AI)", "Auto"], key="stype")
        with col3:
            max_results = st.number_input("Max Results", 5, 50, 12, key="maxres")

        story = st.text_area(
            "Describe what you're looking for:",
            placeholder=f"As a {persona}, I need data on youth depression trends ...",
            height=90, key="story_in",
        )
        submitted = st.form_submit_button("🔍 Search Databases", type="primary")

    if submitted and story:
        search_org = None if org == "All" else org

        with st.spinner("🤖 Searching..."):
            if search_type in ("Keyword", "Auto") and OPENAI_API_KEY:
                queries = optimize_keywords(story, persona)
                with st.expander("🔎 Optimised keyword queries"):
                    for q in queries:
                        st.markdown(f"- `{q}`")
            else:
                queries = [story]

            results: List[Dict] = []
            seen: set = set()
            orgs_to_search = [org] if org != "All" else ["CDC", "SAMHSA"]

            for q in queries:
                for o in orgs_to_search:
                    for ds in socrata_search(q, o, max_results):
                        if ds["uid"] not in seen:
                            seen.add(ds["uid"])
                            results.append(ds)
                if len(results) >= max_results:
                    break

            results = results[:max_results]
            st.session_state.search_results = results
            st.session_state.search_history.append({
                "query": story, "persona": persona, "org": org,
                "ts": datetime.now(), "count": len(results),
            })

        if results:
            st.success(f"✅ Found {len(results)} datasets")
            if OPENAI_API_KEY:
                with st.spinner("Generating persona insights..."):
                    insight = ai_insights(results, persona, story)
                st.markdown(f"""<div class='data-card'>
<h4>🎯 Insights for {persona}</h4><p>{insight}</p></div>""", unsafe_allow_html=True)
        else:
            st.warning("No datasets found. Try different terms.")

    if st.session_state.search_results:
        st.divider()
        st.subheader(f"📊 Results ({len(st.session_state.search_results)} datasets)")

        for i, ds in enumerate(st.session_state.search_results):
            uid, name, ds_org = ds["uid"], ds["name"], ds["org"]
            already = uid in st.session_state.ingested

            with st.expander(f"{i+1}. {name}" + (" ✅" if already else ""), expanded=i < 3):
                st.markdown(f"""<div class='result-card'>
<b>{name}</b> &nbsp;|&nbsp; <code>{uid}</code> &nbsp;|&nbsp; {ds_org}<br>
<small>{ds.get('description','')}</small></div>""", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    n_rows = st.slider("Rows to fetch", 100, 2000, 500, 100, key=f"rows_{uid}")
                with c2:
                    auto_index = st.checkbox("Auto-index for RAG", value=True, key=f"idx_{uid}")
                with c3:
                    st.markdown(f"[Open source ↗]({ds.get('link','')})")

                if not already:
                    if st.button(f"⬇️ Ingest Dataset", key=f"ing_{uid}", type="primary"):
                        with st.spinner(f"Fetching {name}..."):
                            df = socrata_fetch(uid, ds_org, n_rows)
                        if df.empty:
                            st.error("Could not fetch data.")
                        else:
                            st.session_state.ingested[uid] = {
                                "df": df, "name": name, "org": ds_org,
                                "at": datetime.now(), "rows": len(df),
                            }
                            st.session_state.viz_datasets[uid] = {
                                "df": df, "name": name, "is_sampled": len(df) < n_rows,
                                "total_rows": len(df),
                            }
                            if auto_index:
                                with st.spinner("Indexing for RAG..."):
                                    new_chunks = rag_embed(df_to_chunks(df, name, uid))
                                    # Remove old chunks for this uid
                                    st.session_state.rag_index = [
                                        c for c in st.session_state.rag_index if c["uid"] != uid
                                    ]
                                    st.session_state.rag_index.extend(new_chunks)
                            st.success(f"✅ Ingested **{name}** ({len(df):,} rows, "
                                       f"{len(st.session_state.rag_index)} total RAG chunks)")
                            st.rerun()
                else:
                    st.success("✅ Already ingested — see AI Visualizations or Ask Questions tabs.")
                    if st.button("🔄 Re-ingest", key=f"reing_{uid}"):
                        with st.spinner("Re-fetching..."):
                            df = socrata_fetch(uid, ds_org, n_rows)
                        if not df.empty:
                            st.session_state.ingested[uid] = {
                                "df": df, "name": name, "org": ds_org,
                                "at": datetime.now(), "rows": len(df),
                            }
                            st.session_state.viz_datasets[uid] = {
                                "df": df, "name": name, "is_sampled": False, "total_rows": len(df),
                            }
                            if auto_index:
                                new_chunks = rag_embed(df_to_chunks(df, name, uid))
                                st.session_state.rag_index = [
                                    c for c in st.session_state.rag_index if c["uid"] != uid
                                ]
                                st.session_state.rag_index.extend(new_chunks)
                            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — AI Visualizations
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("AI-Powered Data Visualizations")

    if not st.session_state.viz_datasets:
        st.info("Ingest a dataset from the Search & Ingest tab first.")
    else:
        sel_uid = st.selectbox(
            "Select dataset to visualize:",
            list(st.session_state.viz_datasets.keys()),
            format_func=lambda u: st.session_state.viz_datasets[u]["name"],
            key="viz_sel",
        )
        if sel_uid:
            vdata = st.session_state.viz_datasets[sel_uid]
            df    = vdata["df"]
            dname = vdata["name"]
            total = vdata.get("total_rows", len(df))

            if vdata.get("is_sampled"):
                st.warning(f"⚠️ Working with {len(df):,} rows out of {total:,} total "
                           f"({len(df)/max(total,1)*100:.0f}% sample)")
            else:
                st.success(f"Loaded complete dataset: {len(df):,} rows × {len(df.columns)} columns")

            st.divider()
            st.markdown("## 🤖 AI-Powered Analysis")

            cache_key = f"{sel_uid}_{len(df)}"

            col_r, col_regen = st.columns([4, 1])
            with col_regen:
                if st.button("🔄 Regenerate", key=_k("regen")):
                    st.session_state.ai_cache.pop(cache_key, None)
                    st.rerun()

            if not OPENAI_API_KEY:
                st.warning("⚠️ Set OPENAI_API_KEY in Streamlit secrets to enable AI analysis.")
                analysis = None
            elif cache_key not in st.session_state.ai_cache:
                with st.spinner("🧠 AI is analysing your dataset..."):
                    analysis = analyze_dataset_with_ai(df, dname, persona)
                    if "error" not in analysis:
                        st.session_state.ai_cache[cache_key] = analysis
                    else:
                        st.error(f"AI analysis failed: {analysis.get('error')}")
                        analysis = None
            else:
                analysis = st.session_state.ai_cache[cache_key]

            if analysis:
                # Summary & findings
                st.markdown(f"<div class='data-card'><b>📋 Summary</b><br>{analysis.get('summary','')}</div>",
                            unsafe_allow_html=True)
                findings = analysis.get("key_findings", [])
                if findings:
                    st.markdown("**🔍 Key Findings:**")
                    for f in findings:
                        st.markdown(f"- {f}")

                # Charts
                vizs = analysis.get("visualizations", [])
                if vizs:
                    st.divider()
                    st.markdown("### 📈 Recommended Visualizations")
                    cols = st.columns(min(2, len(vizs)))
                    for idx, rec in enumerate(vizs):
                        with cols[idx % 2]:
                            st.markdown(f"**{rec.get('title','')}**")
                            st.caption(rec.get("description", ""))
                            fig = make_chart(rec, df.copy())
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key=_k("vc"))
                            else:
                                st.info("Could not render this chart with the current data.")

            # Manual chart builder
            st.divider()
            st.markdown("### 🛠️ Manual Chart Builder")
            num_cols = df.select_dtypes(include="number").columns.tolist()
            cat_cols = df.select_dtypes(include="object").columns.tolist()
            all_cols = list(df.columns)

            c1, c2, c3, c4 = st.columns(4)
            with c1: chart_type = st.selectbox("Chart type", ["bar","line","scatter","histogram","pie"], key=_k())
            with c2: x_col = st.selectbox("X axis", [None] + all_cols, key=_k())
            with c3: y_col = st.selectbox("Y axis", [None] + num_cols, key=_k())
            with c4: clr   = st.selectbox("Color by", [None] + cat_cols, key=_k())

            if st.button("📊 Generate Chart", key=_k()):
                fig = make_chart({"type": chart_type, "x": x_col, "y": y_col,
                                  "color": clr, "title": f"{chart_type.title()} — {dname}"},
                                 df.copy())
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=_k())
                else:
                    st.warning("Could not render chart. Try different column selections.")

            # Raw data preview
            with st.expander("📋 Raw data preview"):
                st.dataframe(df.head(100), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Ask Questions (AI / RAG)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Ask Questions About Your Data")

    st.info("""
**RAG mode:** your question is answered using the datasets you ingested and indexed.
Turn off RAG to ask general mental-health knowledge questions.
    """)

    question = st.text_area(
        "Your Question:",
        placeholder=f"As a {persona}, what are the key trends in the ingested datasets?",
        height=90, key="qa_q",
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        answer_style = st.selectbox("Answer Style",
                                    ["Detailed", "Concise", "Technical", "Executive Summary"],
                                    key="qa_style")
    with c2:
        include_examples = st.checkbox("Include Examples", value=True, key="qa_ex")
    with c3:
        include_refs = st.checkbox("Add References", value=False, key="qa_refs")

    # Dataset filter
    avail = [(st.session_state.ingested[u]["name"], u)
             for u in st.session_state.ingested]
    ds_filter_uid = None
    if avail:
        choices = ["All ingested datasets"] + [n for n, _ in avail]
        choice = st.selectbox("Limit to a specific dataset:", choices, key="qa_ds_filter")
        if choice != "All ingested datasets":
            ds_filter_uid = next(u for n, u in avail if n == choice)

    rag_mode = st.checkbox("🔍 Use my ingested/indexed data (RAG mode)", value=True, key="qa_rag")

    if st.button("🤖 Get AI Answer", type="primary", key="qa_btn"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif rag_mode and not st.session_state.rag_index:
            st.warning("No datasets indexed yet. Ingest and index at least one dataset first.")
        else:
            with st.spinner("🧠 Generating answer..."):
                if rag_mode:
                    idx = st.session_state.rag_index
                    if ds_filter_uid:
                        idx = [c for c in idx if c["uid"] == ds_filter_uid]
                    top = rag_retrieve(question, idx)
                    extra = ""
                    if include_examples:
                        extra += " Include concrete examples from the data."
                    if include_refs:
                        extra += " Add suggested references or data sources at the end."
                    answer = rag_answer(question, top, persona, answer_style + extra)
                    sources = list({c["name"]: c for c in top}.values())
                    st.session_state.qa_history.append({
                        "q": question, "a": answer, "persona": persona,
                        "style": answer_style, "sources": sources, "rag": True,
                    })
                else:
                    sys_prompt = (
                        f"You are a mental health data expert answering for a {persona}. "
                        "Give an evidence-based answer using general public health knowledge."
                    )
                    answer = chat(question, system=sys_prompt)
                    st.session_state.qa_history.append({
                        "q": question, "a": answer, "persona": persona,
                        "style": answer_style, "sources": [], "rag": False,
                    })

    # History (newest first)
    for entry in reversed(st.session_state.qa_history):
        st.markdown(f"**Q ({entry['persona']}, {entry['style']}):** {entry['q']}")
        st.markdown(entry["a"])
        if entry.get("sources"):
            with st.expander("📄 Retrieved data sources"):
                for s in entry["sources"]:
                    st.caption(f"• {s['name']} (rows {s.get('rows','?')})")
                    st.text(s["text"][:300] + "...")
        mode_tag = "🔍 RAG" if entry.get("rag") else "🌐 General"
        st.caption(f"{mode_tag} | {entry['style']}")
        st.divider()

    if st.session_state.qa_history:
        if st.button("🗑️ Clear history", key="qa_clear"):
            st.session_state.qa_history = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Analytics Dashboard")
    st.markdown("### 📊 Session-Wide Analytics")

    n_ds  = len(st.session_state.ingested)
    n_chk = len(st.session_state.rag_index)
    n_rows_total = sum(v["rows"] for v in st.session_state.ingested.values())
    n_searches   = len(st.session_state.search_history)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ingested Datasets", n_ds)
    c2.metric("Total Records",     f"{n_rows_total:,}")
    c3.metric("RAG Chunks",        f"{n_chk:,}")
    c4.metric("Searches Run",      n_searches)

    if n_ds > 0:
        st.divider()
        st.markdown("### 📁 Ingested Datasets")
        rows_by_org: Dict[str, int] = {}
        for v in st.session_state.ingested.values():
            rows_by_org[v["org"]] = rows_by_org.get(v["org"], 0) + v["rows"]

        c1, c2 = st.columns(2)
        with c1:
            if rows_by_org:
                fig = px.bar(x=list(rows_by_org.keys()), y=list(rows_by_org.values()),
                             title="Records by Organisation", template="plotly_dark",
                             labels={"x": "Org", "y": "Rows"})
                st.plotly_chart(fig, use_container_width=True, key=_k())
        with c2:
            if rows_by_org:
                fig = px.pie(names=list(rows_by_org.keys()), values=list(rows_by_org.values()),
                             title="Record Distribution", hole=0.3, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True, key=_k())

        st.markdown("### 📋 Dataset Inventory")
        inv = []
        for uid, v in st.session_state.ingested.items():
            inv.append({"Name": v["name"], "Org": v["org"], "Rows": v["rows"],
                        "Columns": len(v["df"].columns),
                        "Ingested": v["at"].strftime("%H:%M:%S")})
        st.dataframe(pd.DataFrame(inv), use_container_width=True)

    if n_searches > 0:
        st.divider()
        st.markdown("### 🔍 Search History")
        hist = []
        for h in st.session_state.search_history:
            hist.append({"Query": h["query"][:60], "Persona": h["persona"],
                         "Org": h["org"], "Results": h["count"],
                         "Time": h["ts"].strftime("%H:%M")})
        st.dataframe(pd.DataFrame(hist), use_container_width=True)

    if n_ds == 0 and n_searches == 0:
        st.info("Ingest some datasets to see analytics here.")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Export & Share
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("💾 Export & Share")
    st.markdown("### Export Datasets")

    if not st.session_state.ingested:
        st.info("Ingest datasets first to export them.")
    else:
        ds_options = {uid: v["name"] for uid, v in st.session_state.ingested.items()}
        sel_exp_uid = st.selectbox("Select dataset to export:",
                                   list(ds_options.keys()),
                                   format_func=lambda u: ds_options[u],
                                   key="exp_sel")
        if sel_exp_uid:
            v = st.session_state.ingested[sel_exp_uid]
            df_exp = v["df"]
            st.info(f"**{v['name']}** | {v['org']} | {v['rows']:,} rows | {len(df_exp.columns)} columns")

            c1, c2 = st.columns(2)
            with c1:
                fmt = st.selectbox("Export Format", ["CSV", "JSON", "Parquet"], key="exp_fmt")
            with c2:
                max_exp = st.number_input("Max rows", 100, 50000, min(5000, v["rows"]), key="exp_maxr")

            df_out = df_exp.head(int(max_exp))
            fname  = v["name"].replace(" ", "_")

            if fmt == "CSV":
                st.download_button("⬇️ Download CSV",
                                   df_out.to_csv(index=False).encode(),
                                   file_name=f"{fname}.csv", mime="text/csv", key=_k())
            elif fmt == "JSON":
                st.download_button("⬇️ Download JSON",
                                   df_out.to_json(orient="records", indent=2).encode(),
                                   file_name=f"{fname}.json", mime="application/json", key=_k())
            elif fmt == "Parquet":
                buf = io.BytesIO()
                df_out.to_parquet(buf, index=False)
                st.download_button("⬇️ Download Parquet", buf.getvalue(),
                                   file_name=f"{fname}.parquet",
                                   mime="application/octet-stream", key=_k())

            st.divider()
            st.markdown("### Share Analysis")
            st.caption("Copy this URL to share the platform (data is not included — recipient must ingest their own).")
            st.code("https://share.streamlit.io/sriya19/alfa8-final-mental-health-platform/master/demo.py")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Manage Data
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("⚙️ Manage Data")

    if not st.session_state.ingested:
        st.info("No datasets ingested yet.")
    else:
        st.markdown("### 📋 Ingested Datasets")
        for uid, v in list(st.session_state.ingested.items()):
            c1, c2, c3 = st.columns([4, 1, 1])
            with c1:
                chunks_for = sum(1 for c in st.session_state.rag_index if c["uid"] == uid)
                st.markdown(f"**{v['name']}** | {v['org']} | {v['rows']:,} rows | "
                            f"{chunks_for} RAG chunks")
            with c2:
                if st.button("🗑️ Delete", key=f"del_{uid}"):
                    del st.session_state.ingested[uid]
                    st.session_state.viz_datasets.pop(uid, None)
                    st.session_state.rag_index = [
                        c for c in st.session_state.rag_index if c["uid"] != uid
                    ]
                    st.session_state.ai_cache = {
                        k: v for k, v in st.session_state.ai_cache.items()
                        if not k.startswith(uid)
                    }
                    st.rerun()
            with c3:
                if st.button("🔄 Re-index", key=f"ridx_{uid}"):
                    with st.spinner("Re-indexing..."):
                        df_ri = v["df"]
                        new_chunks = rag_embed(df_to_chunks(df_ri, v["name"], uid))
                        st.session_state.rag_index = [
                            c for c in st.session_state.rag_index if c["uid"] != uid
                        ]
                        st.session_state.rag_index.extend(new_chunks)
                    st.success(f"Re-indexed {len(new_chunks)} chunks.")
            st.divider()

    st.markdown("### 🧩 RAG Index Status")
    st.metric("Total indexed chunks", len(st.session_state.rag_index))
    if st.session_state.rag_index:
        by_ds: Dict[str, int] = {}
        for c in st.session_state.rag_index:
            by_ds[c["name"]] = by_ds.get(c["name"], 0) + 1
        st.dataframe(pd.DataFrame([{"Dataset": k, "Chunks": v}
                                   for k, v in by_ds.items()]),
                     use_container_width=True)

    if st.button("🚨 Clear ALL data", type="secondary", key="clear_all"):
        for key in ["ingested", "rag_index", "viz_datasets", "search_results",
                    "qa_history", "ai_cache", "search_history"]:
            st.session_state[key] = {} if key in ("ingested", "viz_datasets", "ai_cache") else []
        st.success("All session data cleared.")
        st.rerun()
