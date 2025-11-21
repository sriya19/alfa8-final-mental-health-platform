"""
Maryland Mental Health Data Platform - Complete Production Version
All features intact - Full 1600+ lines with all fixes applied
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
import asyncio
import httpx
from dotenv import load_dotenv
import uuid

from backend_api import (
    call_backend, check_backend_health, ingest_dataset, index_dataset,
    preview_dataset, search_datasets_semantic, search_datasets_keyword,
    answer_question, get_rag_status, get_dataset_info, get_corpus_text
)


# Load environment variables
load_dotenv()
load_dotenv('../.env')

# Import our utility modules
from db_utils import (
    sql_df, test_connection, get_last_error, get_database_stats,
    get_maryland_corpus_stats, get_maryland_corpus_by_source, 
    get_maryland_corpus_by_location, get_maryland_corpus_indexed_stats,
    get_ingested_datasets_stats, get_chunks_stats, search_maryland_corpus_text
)


# ===========================
# Configuration
# ===========================
st.set_page_config(
    page_title="🧠 Maryland Mental Health Data Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DEFAULT_ORG = os.getenv("DEFAULT_ORG", "CDC")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

# Personas
PERSONAS = [
    "Public health researcher",
    "Policy maker",
    "Clinician",
    "Epidemiologist",
    "Data analyst"
]

# Enhanced CSS with better contrast and visibility
st.markdown("""
<style>
    /* Main theme with gradient background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Main content area styling */
    [data-testid="stMain"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 20px;
        margin: 10px;
    }
    
    /* Headers in main area */
    [data-testid="stMain"] h1, 
    [data-testid="stMain"] h2, 
    [data-testid="stMain"] h3 {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Sidebar styling with dark theme */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1e 0%, #1a1a2e 100%);
        border-right: 2px solid #667eea;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4 {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Sidebar metrics with gradient backgrounds */
    section[data-testid="stSidebar"] [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.4);
        margin-bottom: 10px;
        backdrop-filter: blur(10px);
    }
    
    section[data-testid="stSidebar"] [data-testid="metric-container"] label {
        color: #a8b2ff !important;
        font-weight: 600;
    }
    
    section[data-testid="stSidebar"] [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.5em;
        font-weight: bold;
    }
    
    /* Success/Error/Warning messages styling */
    .stSuccess {
        background: linear-gradient(135deg, rgba(132, 250, 176, 0.2) 0%, rgba(143, 211, 244, 0.2) 100%);
        color: #84fab0;
        border: 1px solid #84fab0;
        border-radius: 10px;
        padding: 10px;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.2) 0%, rgba(255, 71, 87, 0.2) 100%);
        color: #ff6b6b;
        border: 1px solid #ff6b6b;
        border-radius: 10px;
        padding: 10px;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.2) 0%, rgba(255, 154, 0, 0.2) 100%);
        color: #ffc107;
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Main content cards with dark theme */
    .data-card {
        background: linear-gradient(135deg, rgba(40, 40, 60, 0.95) 0%, rgba(50, 50, 70, 0.95) 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 10px 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
        color: #ffffff;
        backdrop-filter: blur(10px);
    }
    
    .data-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
        border-color: rgba(102, 126, 234, 0.6);
    }
    
    .data-card h4 {
        color: #a8b2ff;
    }
    
    /* Enhanced button styling with gradient */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Sampling info box */
    .sampling-info {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 154, 0, 0.1) 100%);
        border: 1px solid rgba(255, 193, 7, 0.5);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===========================
# Session State Management - COMPLETE
# ===========================
if "results" not in st.session_state:
    st.session_state.results = []
if "last_query" not in st.session_state:
    st.session_state.last_query = {}
if "ingested_datasets" not in st.session_state:
    st.session_state.ingested_datasets = set()
if "indexed_datasets" not in st.session_state:
    st.session_state.indexed_datasets = set()
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "llm_cost" not in st.session_state:
    st.session_state.llm_cost = 0.0
if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = PERSONAS[0]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_error" not in st.session_state:
    st.session_state.last_error = ""
if "viz_datasets" not in st.session_state:
    st.session_state.viz_datasets = {}
if "ai_analysis_cache" not in st.session_state:
    st.session_state.ai_analysis_cache = {}
if "auto_viz_triggered" not in st.session_state:
    st.session_state.auto_viz_triggered = set()
if "dataset_names" not in st.session_state:
    st.session_state.dataset_names = {}
if "chart_id_counter" not in st.session_state:
    st.session_state.chart_id_counter = 0

# ===========================
# Helper function for unique keys
# ===========================
def get_unique_key(prefix="key"):
    """Generate unique key to avoid duplicate element IDs"""
    st.session_state.chart_id_counter += 1
    return f"{prefix}_{st.session_state.chart_id_counter}_{uuid.uuid4().hex[:8]}"

# ===========================
# LLM Functions with Cost Tracking
# ===========================

async def call_openai(prompt: str, system_prompt: str = None, temperature: float = 0.3):
    """Call OpenAI API with cost tracking"""
    if not OPENAI_API_KEY:
        return None, 0.0
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": OPENAI_MODEL,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 2000
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                
                cost = (prompt_tokens * 0.00015 + completion_tokens * 0.0006) / 1000
                
                st.session_state.llm_cost += cost
                logger.info(f"OpenAI usage: {prompt_tokens} prompt, {completion_tokens} completion, cost: ${cost:.4f}")
                
                return data["choices"][0]["message"]["content"], cost
    except Exception as e:
        logger.error(f"OpenAI call failed: {e}")
        st.session_state.last_error = str(e)
    
    return None, 0.0
async def generate_keyword_queries(user_story: str, persona: str) -> List[str]:
    """
    Use LLM to turn a natural-language user story into 2–4 optimized catalog keyword queries.
    Falls back to [user_story] if LLM fails.
    """
    if not OPENAI_API_KEY:
        return [user_story]

    system_prompt = """You are an expert data librarian for public health datasets.
You help translate plain English user stories into optimized keyword queries for the CDC / SAMHSA / Maryland open data catalogs.

Rules:
- Output 2–4 short keyword queries.
- No explanations, no extra text.
- Output MUST be valid JSON: a simple list of strings.
- Each string should be a realistic search like 'BRFSS mental health 2011' or 'drug overdose Maryland county rates'.
"""

    prompt = f"""
Persona: {persona}

User story:
\"\"\"{user_story}\"\"\"

Task:
Return 2–4 optimized keyword queries that will work well when searching the CDC / SAMHSA / Maryland open data catalogs.

Examples of good queries:
- "BRFSS mental health 2011"
- "drug overdose death rates county Maryland"
- "youth depression prevalence state-level"

Now return ONLY JSON, like:
["query 1", "query 2", "query 3"]
"""

    try:
        text, _ = await call_openai(prompt, system_prompt=system_prompt, temperature=0.3)
        if not text:
            return [user_story]

        # Strip code fences if they appear
        txt = text.strip()
        if txt.startswith("```"):
            parts = txt.split("```")
            if len(parts) >= 2:
                txt = parts[1]
                if txt.strip().lower().startswith("json"):
                    txt = txt.strip()[4:]
        txt = txt.strip()

        queries = json.loads(txt)

        # Basic validation
        if not isinstance(queries, list) or not queries:
            return [user_story]

        cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        return cleaned or [user_story]

    except Exception as e:
        logger.error(f"Keyword LLM rewrite failed: {e}")
        return [user_story]

async def generate_persona_insight(data: Any, persona: str, context: str = ""):
    """Generate persona-specific insights using LLM"""
    
    system_prompt = f"""You are a mental health data analyst providing insights for a {persona}.
    
    Tailor your response based on the persona:
    - Public health researcher: Focus on statistical significance, methodology, research implications
    - Policy maker: Focus on actionable recommendations, funding needs, population impact
    - Clinician: Focus on treatment implications, patient care, clinical guidelines
    - Epidemiologist: Focus on disease patterns, risk factors, population health metrics
    - Data analyst: Focus on data quality, trends, correlations, technical insights"""
    
    data_summary = str(data)[:2000] if data else "No data available"
    
    prompt = f"""Analyze this mental health data and provide 3-4 key insights for a {persona}:
    
    Context: {context}
    Data: {data_summary}
    
    Format as bullet points with specific recommendations."""
    
    insight, cost = await call_openai(prompt, system_prompt, temperature=0.5)
    return insight or "Unable to generate insights", cost

async def analyze_dataset_with_ai(df: pd.DataFrame, dataset_name: str, is_sampled: bool = False, 
                                  sample_size: int = None, total_size: int = None) -> Dict[str, Any]:
    """
    AI-powered dataset analysis with visualization recommendations
    Fixed to handle dict/list columns without raising 'unhashable type: dict'
    """
    if not OPENAI_API_KEY:
        return {"error": "OpenAI API key not configured"}
    
    try:
        # Prepare dataset summary
        summary = {
            "name": dataset_name,
            "shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
            "is_sampled": is_sampled,
            "sample_size": sample_size or df.shape[0],
            "total_size": total_size or df.shape[0],
            "columns": []
        }
        
        # Add sampling context
        sampling_context = ""
        if is_sampled:
            sampling_context = f"\nIMPORTANT: This is a SAMPLE of {sample_size:,} rows from a total dataset of {total_size:,} rows."
        
        # Add column information
        for col in df.columns[:30]:
            series = df[col]

            # ---- SAFE null_count ----
            try:
                null_count = int(series.isnull().sum())
            except Exception:
                try:
                    null_count = int(series.isna().sum())
                except Exception:
                    null_count = None

            # ---- SAFE unique_count (avoid 'unhashable type: dict') ----
            try:
                unique_count = int(series.nunique(dropna=True))
            except Exception:
                # Any TypeError / unhashable / weird object -> just mark as unknown
                unique_count = None

            col_info = {
                "name": col,
                "type": str(series.dtype),
                "null_count": null_count,
                "unique_count": unique_count,
                "sample_values": []
            }

            # Sample values – filter out dict/list/set etc.
            try:
                non_null = series.dropna()

                def _is_simple(v):
                    return not isinstance(v, (dict, list, set))

                if len(non_null) > 0:
                    simple_series = non_null[non_null.map(_is_simple)]

                    if len(simple_series) == 0:
                        col_info["sample_values"] = ["Complex type"]
                    else:
                        if series.dtype in ["object", "string"]:
                            try:
                                samples = (
                                    simple_series.value_counts()
                                    .head(3)
                                    .index
                                    .tolist()
                                )
                            except TypeError:
                                samples = simple_series.astype(str).head(3).tolist()
                        else:
                            samples = simple_series.head(3).tolist()

                        col_info["sample_values"] = [str(s)[:50] for s in samples]
            except Exception:
                col_info["sample_values"] = ["Complex type"]

            summary["columns"].append(col_info)
        
        # Create prompt for AI with explicit instructions
        prompt = f"""Analyze this mental health dataset and suggest visualizations.

Dataset: {dataset_name}
Shape: {summary['shape']}{sampling_context}

Columns (first 30):
{json.dumps(summary['columns'], indent=2)[:3000]}

Provide a JSON response with EXACTLY this structure:
{{
    "narrative": "A 2-3 sentence analysis of what this dataset contains and its potential value for mental health research",
    "sampling_note": "A note about sampling limitations if this is sampled data (or null if not sampled)",
    "visualizations": [
        {{
            "type": "chart_type",
            "columns": {{"x_column": "column_name", "y_column": "column_name"}},
            "reasoning": "One sentence explaining why this visualization would be valuable",
            "insights": "One sentence about what patterns or trends this might reveal"
        }}
    ]
}}

Requirements:
1. Include exactly 3-5 visualizations
2. Chart types must be one of: histogram, bar, line, scatter, heatmap, box, pie
3. Column names must exactly match columns from the dataset
4. Each visualization must have non-empty reasoning and insights
5. For heatmap type, include "value_column" in columns object

Return ONLY valid JSON, no markdown formatting or code blocks."""
        
        # Call OpenAI
        response_text, cost = await call_openai(prompt, temperature=0.3)
        
        if not response_text:
            return {"error": "Failed to get AI response"}
        
        # Parse response
        response_text = response_text.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        analysis = json.loads(response_text)
        
        # Validate and fix visualizations
        valid_visualizations = []
        for viz in analysis.get("visualizations", []):
            if viz.get("type") and viz.get("columns"):
                if not viz.get("reasoning"):
                    viz["reasoning"] = f"This {viz['type']} chart shows relationships in the data"
                if not viz.get("insights"):
                    viz["insights"] = "May reveal patterns and trends in the dataset"
                valid_visualizations.append(viz)
        
        analysis["visualizations"] = valid_visualizations
        analysis['is_sampled'] = is_sampled
        analysis['cost'] = cost
        
        return analysis
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI response as JSON: {e}")
        # Return a default analysis
        return {
            "narrative": f"This dataset contains {df.shape[0]} rows and {df.shape[1]} columns of mental health data.",
            "visualizations": [
                {
                    "type": "histogram",
                    "columns": {"x_column": df.columns[0] if len(df.columns) > 0 else "column1"},
                    "reasoning": "Shows the distribution of values in the first column",
                    "insights": "Helps identify data patterns and outliers"
                }
            ],
            "is_sampled": is_sampled,
            "cost": 0
        }
    except Exception as e:
        logger.error(f"AI analysis failed: {e}")
        return {"error": str(e)}

async def extract_table_from_corpus(corpus_text: str, dataset_name: str):
    """
    Use LLM to extract a tabular dataset (rows/columns) from unstructured corpus text.
    Returns (DataFrame or None, error_message or None).
    """
    if not OPENAI_API_KEY:
        return None, "OpenAI API key not configured"

    # Keep prompt size reasonable
    text_snippet = corpus_text[:6000]

    system_prompt = (
        "You are a data engineer turning unstructured mental health corpus text "
        "into a clean tabular dataset suitable for analysis in pandas."
    )

    prompt = f"""
You are given unstructured text from a mental health–related corpus for a dataset:

Dataset name: {dataset_name}

TEXT:
\"\"\"{text_snippet}\"\"\"

Your task:

1. Identify a useful, reasonably sized table that could be created from this text.
2. Infer clear, concise column names (e.g., "year", "state", "age_group", "measure", "value", "unit").
3. Extract as many rows as you can reliably infer.
4. Only include numeric values where they clearly belong (e.g., percentages, counts, rates).

Return ONLY valid JSON in this exact format (no markdown, no extra text):

[
  {{"column1": value1, "column2": value2, ...}},
  {{"column1": value3, "column2": value4, ...}}
]

Where each object is one row. Do not wrap this in any additional object.
"""

    try:
        response_text, _ = await call_openai(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
        )

        if not response_text:
            return None, "LLM returned empty response"

        # Strip ```json fences if present
        txt = response_text.strip()
        if txt.startswith("```"):
            parts = txt.split("```")
            if len(parts) >= 2:
                txt = parts[1]
                if txt.strip().startswith("json"):
                    txt = txt.strip()[4:]
        txt = txt.strip()

        import json
        rows = json.loads(txt)

        if not isinstance(rows, list) or not rows:
            return None, "LLM did not return a non-empty JSON list"

        df = pd.DataFrame(rows)
        if df.empty:
            return None, "Extracted table is empty"

        return df, None

    except Exception as e:
        return None, f"Failed to extract table: {e}"

# ===========================
# Enhanced Visualization Functions - COMPLETE VERSION
# ===========================

def _resolve_column(col_name: Optional[str], df: pd.DataFrame) -> Optional[str]:
    """Return the real dataframe column name, case-insensitive."""
    if not col_name:
        return None
    if col_name in df.columns:
        return col_name
    lower_map = {c.lower(): c for c in df.columns}
    return lower_map.get(col_name.lower())

def create_plotly_from_recommendation(viz_rec: Dict, df: pd.DataFrame) -> Optional[go.Figure]:
    """Generate Plotly chart from AI recommendation - Fixed and Complete"""
    try:
        chart_type = viz_rec.get('type', '').lower()
        columns = viz_rec.get('columns', {})
        
        # Use case-insensitive column resolution
        x_col = _resolve_column(columns.get('x_column'), df)
        y_col = _resolve_column(columns.get('y_column'), df)
        value_col = _resolve_column(columns.get('value_column'), df)
        
        if chart_type == 'histogram':
            if x_col and x_col in df.columns:
                if pd.api.types.is_numeric_dtype(df[x_col]):
                    fig = px.histogram(df, x=x_col, title=f"Distribution of {x_col}", marginal="box")
                    fig.update_layout(template="plotly_dark", height=400)
                    return fig
                else:
                    value_counts = df[x_col].value_counts().head(20)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                                title=f"Distribution of {x_col}")
                    fig.update_layout(template="plotly_dark", height=400, 
                                    xaxis_title=x_col, yaxis_title="Count")
                    return fig
        
        elif chart_type == 'bar':
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                if df[x_col].dtype == 'object' and pd.api.types.is_numeric_dtype(df[y_col]):
                    agg_df = df.groupby(x_col)[y_col].mean().reset_index()
                    agg_df = agg_df.sort_values(y_col, ascending=False).head(20)
                    fig = px.bar(agg_df, x=x_col, y=y_col, title=f"Average {y_col} by {x_col}",
                                color=y_col, color_continuous_scale='Viridis')
                elif pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                    df['binned'] = pd.cut(df[x_col], bins=10)
                    agg_df = df.groupby('binned')[y_col].mean().reset_index()
                    agg_df['binned'] = agg_df['binned'].astype(str)
                    fig = px.bar(agg_df, x='binned', y=y_col, title=f"{y_col} by {x_col} bins")
                else:
                    value_counts = df[x_col].value_counts().head(20)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                                title=f"Count of {x_col}")
                fig.update_layout(template="plotly_dark", height=400)
                return fig
        
        elif chart_type == 'line':
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                df_sorted = df.sort_values(x_col)
                if len(df_sorted) > 1000:
                    df_sorted = df_sorted.sample(n=1000).sort_values(x_col)
                fig = px.line(df_sorted, x=x_col, y=y_col, title=f"{y_col} Over {x_col}",
                             markers=True)
                fig.update_layout(template="plotly_dark", height=400)
                return fig
        
        elif chart_type == 'scatter':
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                    df_plot = df.sample(n=min(1000, len(df)))
                    fig = px.scatter(df_plot, x=x_col, y=y_col, title=f"{y_col} vs {x_col}",
                                   opacity=0.6, trendline="ols" if len(df_plot) < 500 else None)
                    fig.update_layout(template="plotly_dark", height=400)
                    return fig
        
        elif chart_type == 'heatmap':
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                try:
                    if value_col and value_col in df.columns:
                        pivot_df = df.pivot_table(values=value_col, index=y_col, 
                                                  columns=x_col, aggfunc='mean')
                    else:
                        pivot_df = df.pivot_table(index=y_col, columns=x_col, 
                                                  aggfunc='size', fill_value=0)

                    
                    if pivot_df.shape[0] > 50:
                        pivot_df = pivot_df.iloc[:50]
                    if pivot_df.shape[1] > 50:
                        pivot_df = pivot_df.iloc[:, :50]
                    
                    fig = px.imshow(pivot_df, title=f"Heatmap: {y_col} vs {x_col}", 
                                   labels=dict(color="Value"), aspect="auto",
                                   color_continuous_scale='RdBu')
                    fig.update_layout(template="plotly_dark", height=400)
                    return fig
                except:
                    return None
        
        elif chart_type == 'box':
            if y_col and y_col in df.columns:
                if pd.api.types.is_numeric_dtype(df[y_col]):
                    if x_col and x_col in df.columns:
                        if df[x_col].nunique() <= 20:
                            fig = px.box(df, x=x_col, y=y_col, 
                                       title=f"Distribution of {y_col} by {x_col}",
                                       points="outliers")
                        else:
                            fig = px.box(df, y=y_col, title=f"Distribution of {y_col}",
                                       points="outliers")
                    else:
                        fig = px.box(df, y=y_col, title=f"Distribution of {y_col}",
                                   points="outliers")
                    fig.update_layout(template="plotly_dark", height=400)
                    return fig
        
        elif chart_type == 'pie':
            if x_col and x_col in df.columns:
                value_counts = df[x_col].value_counts().head(10)
                fig = px.pie(values=value_counts.values, names=value_counts.index, 
                           title=f"Distribution of {x_col}", hole=0.3)
                fig.update_layout(template="plotly_dark", height=400)
                return fig
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to create chart: {e}")
        return None

def safe_create_visualization(viz_rec: Dict, df: pd.DataFrame, index: int) -> Optional[go.Figure]:
    """
    Safely create a visualization from an AI recommendation.

    Guarantees that, if there is at least one numeric or categorical column,
    we return a non-empty Plotly figure (so the UI will not show the red error).
    """
    chart_type = (viz_rec.get("type") or "").lower()

    # Column lists
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    # 1) Try the strict AI recommendation first
    try:
        fig = create_plotly_from_recommendation(viz_rec, df)
        if fig is not None and len(fig.data) > 0:
            return fig
    except Exception as e:
        logger.error(f"AI viz failed in strict mode: {e}")

    # 2) Robust fallbacks by chart type
    try:
        # --- SCATTER ---
        if chart_type == "scatter" and len(numeric_cols) >= 2:
            x = numeric_cols[0]
            y = numeric_cols[1]
            sample = df.sample(n=min(1000, len(df)))
            fig = px.scatter(sample, x=x, y=y, title=f"{y} vs {x}")
            fig.update_layout(template="plotly_dark", height=400)
            return fig

        # --- BOX ---
        if chart_type == "box" and numeric_cols:
            col = numeric_cols[index % len(numeric_cols)]
            fig = px.box(df, y=col, title=f"Distribution of {col}", points="outliers")
            fig.update_layout(template="plotly_dark", height=400)
            return fig

        # --- HEATMAP ---
        if chart_type == "heatmap":
            # If we have >= 2 numeric columns, use a correlation heatmap
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                fig = px.imshow(
                    corr,
                    title="Correlation Heatmap (numeric columns)",
                    labels=dict(color="Correlation"),
                    color_continuous_scale="RdBu",
                    zmin=-1,
                    zmax=1,
                    text_auto=".2f",
                )
                fig.update_layout(template="plotly_dark", height=400)
                return fig

            # Otherwise fall back to a simple count heatmap for first 2 categoricals
            if len(categorical_cols) >= 2:
                c1, c2 = categorical_cols[:2]
                crosstab = pd.crosstab(df[c1], df[c2])
                fig = px.imshow(
                    crosstab,
                    title=f"Heatmap of {c1} by {c2}",
                    labels=dict(color="Count"),
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(template="plotly_dark", height=400)
                return fig

        # --- HISTOGRAM / BAR (generic fallback for anything numeric) ---
        if numeric_cols:
            col = numeric_cols[index % len(numeric_cols)]
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            fig.update_layout(template="plotly_dark", height=400)
            return fig

        # --- PURE CATEGORICAL FALLBACK (if no numeric columns at all) ---
        if categorical_cols:
            col = categorical_cols[index % len(categorical_cols)]
            value_counts = df[col].value_counts().head(15)
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Counts of {col}",
                labels={"x": col, "y": "Count"},
            )
            fig.update_layout(template="plotly_dark", height=400)
            return fig

    except Exception as e:
        logger.error(f"Visualization error in fallback: {e}")

    # If we truly have nothing to plot, return None so the UI shows the error.
    return None


def create_all_visualizations(df: pd.DataFrame, dataset_name: str) -> List[go.Figure]:
    """Create comprehensive interactive visualizations for a dataset - COMPLETE VERSION"""
    figures = []
    
    # Get column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Limit for performance
    numeric_cols = numeric_cols[:15]
    categorical_cols = categorical_cols[:10]
    
    colors = px.colors.qualitative.Set3
    
    # 1. HISTOGRAMS for numeric distributions
    for i, col in enumerate(numeric_cols[:4]):
        try:
            fig = px.histogram(
                df, x=col, 
                title=f"Distribution of {col}",
                color_discrete_sequence=[colors[i % len(colors)]],
                marginal="box"  # Add box plot on top
            )
            fig.update_layout(template="plotly_dark", height=400)
            figures.append(fig)
        except:
            pass
    
    # 2. BAR CHARTS for categorical data
    for col in categorical_cols[:3]:
        try:
            value_counts = df[col].value_counts().head(15)
            fig = px.bar(
                x=value_counts.index, y=value_counts.values,
                title=f"Top 15 {col} Categories",
                labels={'x': col, 'y': 'Count'},
                color=value_counts.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(template="plotly_dark", height=400)
            figures.append(fig)
        except:
            pass
    
    # 3. SCATTER PLOTS for numeric relationships
    if len(numeric_cols) >= 2:
        pairs = [(numeric_cols[i], numeric_cols[i+1]) for i in range(min(3, len(numeric_cols)-1))]
        
        for x_col, y_col in pairs:
            try:
                fig = px.scatter(
                    df.sample(n=min(1000, len(df))), x=x_col, y=y_col,
                    title=f"{y_col} vs {x_col}",
                    opacity=0.6,
                    trendline="ols" if len(df) < 1000 else None
                )
                fig.update_layout(template="plotly_dark", height=400)
                figures.append(fig)
            except:
                pass
    
    # 4. BOX PLOTS for outlier detection
    for col in numeric_cols[:3]:
        try:
            fig = px.box(
                df, y=col,
                title=f"Box Plot: {col} (Outlier Detection)",
                points="outliers"
            )
            fig.update_layout(template="plotly_dark", height=400)
            figures.append(fig)
        except:
            pass
    
    # 5. CORRELATION HEATMAP
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols[:10]].corr()
            fig = px.imshow(
                corr_matrix,
                title="Correlation Heatmap",
                labels=dict(color="Correlation"),
                color_continuous_scale='RdBu',
                zmin=-1, zmax=1,
                text_auto='.2f'
            )
            fig.update_layout(template="plotly_dark", height=500)
            figures.append(fig)
        except:
            pass
    
    # 6. TIME SERIES if date column exists
    if date_cols and numeric_cols:
        date_col = date_cols[0]
        for value_col in numeric_cols[:2]:
            try:
                df_sorted = df.sort_values(date_col)
                fig = px.line(
                    df_sorted, x=date_col, y=value_col,
                    title=f"{value_col} Over Time",
                    markers=True
                )
                fig.update_layout(template="plotly_dark", height=400)
                figures.append(fig)
            except:
                pass
    
    # 7. PIE CHART for proportions
    if categorical_cols:
        col = categorical_cols[0]
        try:
            value_counts = df[col].value_counts().head(10)
            fig = px.pie(
                values=value_counts.values, 
                names=value_counts.index,
                title=f"Distribution of {col}",
                hole=0.3
            )
            fig.update_layout(template="plotly_dark", height=400)
            figures.append(fig)
        except:
            pass
    
    # 8. GROUPED BAR CHART if we have categorical + numeric
    if categorical_cols and len(numeric_cols) >= 2:
        try:
            cat_col = categorical_cols[0]
            # Take top categories to avoid clutter
            top_categories = df[cat_col].value_counts().head(8).index
            df_filtered = df[df[cat_col].isin(top_categories)]
            
            fig = go.Figure()
            for num_col in numeric_cols[:3]:
                agg_df = df_filtered.groupby(cat_col)[num_col].mean().reset_index()
                fig.add_trace(go.Bar(
                    name=num_col,
                    x=agg_df[cat_col],
                    y=agg_df[num_col]
                ))
            
            fig.update_layout(
                title=f"Comparison Across {cat_col}",
                barmode='group',
                template="plotly_dark",
                height=400
            )
            figures.append(fig)
        except:
            pass
    
    # 9. VIOLIN PLOT for distribution comparison
    if categorical_cols and numeric_cols:
        try:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            if df[cat_col].nunique() <= 10:
                fig = px.violin(
                    df, x=cat_col, y=num_col,
                    title=f"Distribution of {num_col} by {cat_col}",
                    box=True, points="outliers"
                )
                fig.update_layout(template="plotly_dark", height=400)
                figures.append(fig)
        except:
            pass
    
    # 10. SUNBURST for hierarchical data
    if len(categorical_cols) >= 2:
        try:
            cat1, cat2 = categorical_cols[:2]
            df_grouped = df.groupby([cat1, cat2]).size().reset_index(name='count')
            df_grouped = df_grouped.head(50)  # Limit for performance
            
            fig = px.sunburst(
                df_grouped, path=[cat1, cat2], values='count',
                title=f"Hierarchical View: {cat1} → {cat2}"
            )
            fig.update_layout(template="plotly_dark", height=400)
            figures.append(fig)
        except:
            pass
    
    return figures

# ===========================
# Main Application - COMPLETE WITH ALL FEATURES
# ===========================

def main():
    # Header with gradient
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 15px; margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
        <h1 style='color: white; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
            🧠 Maryland Mental Health Data Platform
        </h1>
        <p style='color: white; text-align: center; margin: 10px 0 0 0; opacity: 0.9;'>
            AI-Powered Analysis • RAG-Enhanced Insights • 15k+ Semantic Chunks • Production Ready
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - COMPLETE
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Organization selector
        org = st.selectbox(
            "🏢 Filter by Organization",
            ["All", "CDC", "SAMHSA", "Maryland"],
            index=0,
            help="""
            🔍 **Keyword Search**: Searches external APIs (All = CDC default)
            🤖 **Semantic Search**: Optional filter (All = search entire corpus)
            💬 **RAG**: Optional filter for context
            """,
            key="org_selector_main"
        )
        
        # Persona selector
        st.markdown("### 👤 Select Your Role")
        selected_persona = st.selectbox(
            "Choose your persona:",
            PERSONAS,
            help="This tailors all insights and recommendations to your specific role",
            key="persona_selector_main"
        )
        st.session_state.selected_persona = selected_persona
        
        # Show persona description
        persona_descriptions = {
            "Public health researcher": "📊 Statistical analysis, research methodology, evidence-based insights",
            "Policy maker": "📋 Actionable recommendations, funding priorities, population impact",
            "Clinician": "🏥 Clinical implications, treatment guidelines, patient care insights",
            "Epidemiologist": "📈 Disease patterns, risk factors, population health metrics",
            "Data analyst": "💻 Technical insights, data quality, trends and correlations"
        }
        
        st.info(persona_descriptions.get(selected_persona, ""))
        
        st.divider()
        
        # Database Statistics - REAL DATA (FIXED)
        st.markdown("### 📊 Database Overview")
        
        db_stats = get_database_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📚 Total Records", f"{db_stats.get('total_records', 0):,}")
            st.metric("🗂️ Unique Datasets", f"{db_stats.get('unique_datasets', 0):,}")
        
        with col2:
            st.metric("📍 Locations", f"{db_stats.get('unique_locations', 0):,}")
            st.metric("🔗 Data Sources", f"{db_stats.get('data_sources', 0):,}")
        
        # Chunks breakdown
        st.markdown("#### 🧩 Semantic Chunks")
        st.metric("Total Chunks", f"{db_stats.get('total_chunks', 0):,}")
        
        # Get chunks stats for corpus breakdown
        try:
            chunks_stats = get_chunks_stats()
            maryland_chunks = chunks_stats.get('maryland_chunks', 0)
            ingested_chunks = chunks_stats.get('ingested_chunks', 0)
            
            if maryland_chunks + ingested_chunks > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"🏛️ Maryland: {maryland_chunks:,}")
                with col2:
                    st.caption(f"📊 Ingested: {ingested_chunks:,}")
                
                if chunks_stats.get('total_chunks', 0) > 0:
                    md_pct = (maryland_chunks / chunks_stats['total_chunks']) * 100
                    st.progress(md_pct / 100)
                    st.caption(f"Maryland corpus: {md_pct:.1f}%")
        except:
            pass
        
        st.divider()
        
        # Additional corpus stats
        try:
            md_stats = get_maryland_corpus_stats()
            if md_stats['total_records'] > 0:
                st.markdown("### 🏛️ Maryland Corpus")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Records", f"{md_stats['total_records']:,}")
                with col2:
                    st.metric("Sources", md_stats['unique_sources'])
                
                indexed_stats = get_maryland_corpus_indexed_stats()
                if indexed_stats['total_chunks'] > 0:
                    st.success(f"✅ {indexed_stats['total_chunks']:,} chunks indexed")
        except:
            pass
        
        st.divider()
        
        # Session metrics
        st.markdown("### 💾 Session Info")
        st.metric("Ingested", len(st.session_state.ingested_datasets))
        st.metric("Indexed", len(st.session_state.indexed_datasets))
        st.metric("LLM Cost", f"${st.session_state.llm_cost:.4f}")
        
        # Diagnostics
        with st.expander("🔍 Diagnostics", expanded=False):
            health = check_backend_health()
            if health['status'] == 'healthy':
                st.success("✅ Backend: Healthy")
            else:
                st.error(f"❌ Backend: {health.get('error', 'Unknown error')}")
            
            if test_connection():
                st.success("✅ Database: Connected")
            else:
                st.error("❌ Database: Disconnected")
            
            if st.session_state.last_error:
                st.warning(f"Last error: {st.session_state.last_error}")
    
    # Main content tabs
    tabs = st.tabs([
        "🔍 Search & Ingest",
        "📊 AI Visualizations",
        "💬 Ask Questions (AI)",
        "📈 Analytics Dashboard",
        "💾 Export & Share",
        "⚙️ Manage Data"
    ])
    
    # Tab 1: Search & Ingest - COMPLETE WITH FIXES
    with tabs[0]:
        st.header("Find and Ingest Relevant Datasets")
        
        # Info about search scope
        try:
            chunks_stats = get_chunks_stats()
            total_chunks = chunks_stats.get('total_chunks', 0)
            if total_chunks > 0:
                st.info(f"💡 **Semantic search will scan {total_chunks:,} chunks** from Maryland corpus + ingested datasets")
        except:
            pass
        
        # Search interface
        with st.form("search_form"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**Searching as:** {selected_persona}")
            
            with col2:
                search_type = st.selectbox(
                    "Search Type",
                    ["Semantic (AI)", "Keyword", "Auto"],
                    help="Semantic uses AI embeddings. Keyword searches external APIs.",
                    key="search_type_selector"
                )
            
            with col3:
                max_results = st.number_input("Max Results", 5, 50, 10, key="max_results_selector")
            
            user_story = st.text_area(
                "Describe what you're looking for:",
                placeholder=f"As a {selected_persona}, I need data on youth depression trends in Baltimore County from 2020-2023...",
                height=100,
                key="user_story_input"
            )
            
            submitted = st.form_submit_button("🔍 Search Databases", type="primary")
        
        if submitted and user_story:
            with st.spinner("🤖 AI is searching for relevant datasets..."):
                
                # Perform search with correct org handling
                if search_type == "Semantic (AI)":
                    # For semantic search, pass None if "All" is selected
                    search_org = None if org == "All" else org
                    results, error = search_datasets_semantic(
                        search_org, user_story, max_results, selected_persona, 
                        filter_ingested_only=False
                    )
                    
                elif search_type == "Keyword":
                    # For keyword search, use CDC as default if "All" is selected
                    search_org = "CDC" if org == "All" else org

                    # --- NEW: LLM-assisted keyword rewriting ---
                    if OPENAI_API_KEY:
                        st.info("✨ Using AI to optimize your keyword search…")
                        try:
                            optimized_queries = asyncio.run(
                                generate_keyword_queries(user_story, selected_persona)
                            )
                        except Exception as e:
                            logger.error(f"generate_keyword_queries failed: {e}")
                            optimized_queries = [user_story]
                    else:
                        optimized_queries = [user_story]

                    # Show what we’ll actually search
                    with st.expander("🔎 Optimized keyword queries", expanded=False):
                        for q in optimized_queries:
                            st.markdown(f"- `{q}`")

                    # Call backend for each query and merge results
                    all_results = []
                    seen_ids = set()
                    error = None

                    for q in optimized_queries:
                        part_results, err = search_datasets_keyword(
                            search_org, q, max_results
                        )

                        if err and not error:
                            error = err  # keep first error

                        if part_results:
                            for r in part_results:
                                # Try to deduplicate by dataset_uid / uid
                                uid = r.get("dataset_uid") or r.get("uid") or r.get("id")
                                key = f"{search_org}:{uid}" if uid else json.dumps(r, sort_keys=True)

                                if key not in seen_ids:
                                    seen_ids.add(key)
                                    all_results.append(r)

                    # Truncate to max_results total
                    results = all_results[:max_results]
                    
                else:  # Auto
                    search_org = None if org == "All" else org
                    results, error = search_datasets_semantic(
                        search_org, user_story, max_results, selected_persona,
                        filter_ingested_only=False
                    )
                    if not results and not error:
                        st.info("No semantic results found. Try keyword search...")
                        search_org = "CDC" if org == "All" else org
                        results, error = search_datasets_keyword(
                            search_org, user_story, max_results
                        )
                
                if error:
                    st.error(f"Search failed: {error}")
                    st.session_state.last_error = error
                else:
                    st.session_state.results = results
                    
                    # Add to search history
                    st.session_state.search_history.append({
                        "query": user_story,
                        "persona": selected_persona,
                        "search_type": f"{search_type} (LLM)" if search_type == "Keyword" and OPENAI_API_KEY else search_type,
                        "org_filter": org,
                        "timestamp": datetime.now(),
                        "results_count": len(results)
                    })

                    
                    if results:
                        st.success(f"✅ Found {len(results)} relevant datasets")
                        
                        # Generate persona-specific insights
                        if OPENAI_API_KEY and len(results) > 0:
                            with st.spinner("Generating insights..."):
                                insight, _ = asyncio.run(generate_persona_insight(
                                    results[:3],
                                    selected_persona,
                                    f"Search results for: {user_story}"
                                ))
                                
                                st.markdown(f"""
                                <div class='data-card'>
                                    <h4>🎯 Insights for {selected_persona}</h4>
                                    <p>{insight}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("No datasets found. Try different search terms or search type.")
        
        # Display search results - COMPLETE WORKING VERSION
        if st.session_state.results:
            st.divider()
            st.subheader(f"📊 Search Results ({len(st.session_state.results)} datasets)")
            
            for i, result in enumerate(st.session_state.results):
                # Get all possible UIDs
                uid = result.get('uid', '')
                dataset_uid = result.get('dataset_uid', uid)  # Fallback to uid
                internal_uid = result.get('internal_uid', uid)
                
                # CRITICAL: Use the right UID for operations
                operation_uid = dataset_uid if dataset_uid else uid
                
                # Determine source type
                source_type = result.get('source', 'unknown')
                corpus_type = result.get('corpus_type', 'unknown')
                
                # Determine badge
                if corpus_type == 'maryland':
                    badge = "🏛️ **Maryland Corpus**"
                    badge_color = "#f093fb"
                elif source_type == 'semantic':
                    badge = "🤖 **Semantic Match**"
                    badge_color = "#667eea"
                elif source_type == 'external_api':
                    badge = "🌐 **External API**"
                    badge_color = "#84fab0"
                else:
                    badge = "📊 **Database**"
                    badge_color = "#84fab0"
                
                # Get display name
                display_name = result.get('name', result.get('title', f"Dataset {operation_uid}"))
                dataset_org = result.get('org', org)
                
                with st.expander(f"{i+1}. {display_name}", expanded=(i < 3)):
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {badge_color}33 0%, {badge_color}11 100%); 
                                padding: 8px 12px; border-radius: 8px; margin-bottom: 10px; 
                                border-left: 3px solid {badge_color};'>
                        {badge}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**Dataset ID:** `{operation_uid}`")
                    st.markdown(f"**Organization:** {dataset_org}")
                    
                    if result.get('link'):
                        st.markdown(f"**Source:** {result['link']}")
                    
                    if result.get('score') or result.get('similarity'):
                        score = result.get('score', result.get('similarity', 0))
                        st.markdown(f"**Relevance Score:** {score:.3f}")
                    
                    desc = result.get('description', 'No description available')
                    st.text_area(
                        "Dataset Description",
                        desc[:1000] if len(desc) > 1000 else desc,
                        height=150,
                        disabled=True,
                        key=f"desc_{i}_{operation_uid}"
                    )
                    
                    # Action buttons - FIXED WITH CORRECT UID HANDLING
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        # Ingest button
                        is_ingested = operation_uid in st.session_state.ingested_datasets
                        can_ingest = result.get('can_ingest', True)  # Default to True
                        
                        if corpus_type == 'maryland' and not can_ingest:
                            st.info("Pre-indexed")
                        elif is_ingested:
                            st.success("✅ Ingested")
                        elif can_ingest:
                            auto_index = st.checkbox(
                                "Auto-index",
                                key=f"auto_{i}_{operation_uid}",
                                help="Automatically create searchable chunks after ingesting"
                            )
                            
                            if st.button(f"💾 Ingest", key=f"ingest_{i}_{operation_uid}"):
                                with st.spinner(f"Ingesting dataset {operation_uid}..."):
                                    # CRITICAL: Use the correct UID
                                    success, message = ingest_dataset(dataset_org, operation_uid, auto_index)
                                    
                                    if success:
                                        st.success(message)
                                        st.session_state.ingested_datasets.add(operation_uid)
                                        st.session_state.dataset_names[operation_uid] = display_name
                                        if auto_index:
                                            st.session_state.indexed_datasets.add(operation_uid)
                                        st.balloons()
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(f"Failed: {message}")
                                        st.session_state.last_error = message
                        else:
                            st.info("Cannot ingest")
                    
                    with col2:
                        # Index button
                        is_indexed = operation_uid in st.session_state.indexed_datasets
                        if is_indexed:
                            st.success("📚 Indexed")
                        elif is_ingested:
                            if st.button(f"📚 Index", key=f"index_{i}_{operation_uid}"):
                                with st.spinner(f"Indexing for RAG..."):
                                    success, message = index_dataset(dataset_org, operation_uid)
                                    
                                    if success:
                                        st.success(message)
                                        st.session_state.indexed_datasets.add(operation_uid)
                                        st.rerun()
                                    else:
                                        st.error(f"Failed: {message}")
                                        st.session_state.last_error = message
                    
                    with col3:
                        # Preview button
                        if st.button(f"👁️ Preview", key=f"preview_{i}_{operation_uid}"):
                            with st.spinner("Loading preview..."):
                                # CRITICAL: Use correct UID
                                df = preview_dataset(dataset_org, operation_uid, 200)
                                if df is not None and len(df) > 0:
                                    if hasattr(df, 'attrs'):
                                        total = df.attrs.get('total_rows', len(df))
                                        if df.attrs.get('is_sampled', False):
                                            st.warning(f"⚠️ Showing {len(df)} of {total:,} total rows")
                                    
                                    st.write(f"**Preview** ({len(df)} rows):")
                                    st.dataframe(df.head(50))
                                else:
                                    st.error("Could not load preview")
                    
                    with col4:
                        # 📊 Visualization / extraction logic
                        if corpus_type == "maryland":
                            # For Maryland corpus (pre-indexed text only), we use LLM to extract a table.
                            if st.button(
                                "📊 Extract & Visualize",
                                key=f"extract_{i}_{operation_uid}",
                            ):
                                with st.spinner("Using AI to extract structured data from Maryland corpus..."):
                                    # Use the internal UID used in the chunks table
                                    corpus_uid = internal_uid or operation_uid

                                    corpus_data, error = get_corpus_text(corpus_uid)
                                    if error or not corpus_data or not corpus_data.get("text"):
                                        st.error(f"Could not load corpus text: {error or 'no text returned'}")
                                    else:
                                        try:
                                            df, err2 = asyncio.run(
                                                extract_table_from_corpus(corpus_data["text"], display_name)
                                            )
                                            if df is None or df.empty:
                                                st.error(err2 or "AI could not extract a useful table from this corpus.")
                                            else:
                                                # Store for visualization tab
                                                st.session_state.viz_datasets[corpus_uid] = {
                                                    "df": df,
                                                    "name": display_name + " (AI-extracted)",
                                                    "org": dataset_org,
                                                    "is_sampled": True,
                                                    "total_rows": len(df),
                                                }
                                                st.session_state.auto_viz_triggered.add(corpus_uid)
                                                st.success("✅ Data extracted! Go to AI Visualizations tab")
                                                st.info("Switch to the 'AI Visualizations' tab to explore charts.")
                                        except Exception as e:
                                            st.error(f"Extraction failed: {e}")
                                            st.session_state.last_error = str(e)

                        else:
                            # Regular datasets use normal preview-based visualize
                            if st.button(
                                "📊 Visualize",
                                key=f"viz_{i}_{operation_uid}",
                            ):
                                with st.spinner("Loading for visualization..."):
                                    sample_size = 5000
                                    df = preview_dataset(dataset_org, operation_uid, sample_size)
                                    if df is not None and len(df) > 0:
                                        st.session_state.viz_datasets[operation_uid] = {
                                            "df": df,
                                            "name": display_name,
                                            "org": dataset_org,
                                            "is_sampled": hasattr(df, "attrs")
                                            and df.attrs.get("is_sampled", False),
                                            "total_rows": getattr(df, "attrs", {}).get(
                                                "total_rows", len(df)
                                            ),
                                        }
                                        st.session_state.auto_viz_triggered.add(operation_uid)
                                        st.success("✅ Data loaded! Go to AI Visualizations tab")
                                        st.info("Switch to the 'AI Visualizations' tab to see your data")
                                    else:
                                        st.error("Could not load data for visualization")
                    
                    with col5:
                        link = result.get('link', '')
                        if link:
                            st.link_button("🔗 Source", link)
    
    # Tab 2: AI Visualizations - UPDATED TO ALWAYS GENERATE AI RECS + AI-EXTRACTED HANDLING
    with tabs[1]:
        st.header("AI-Powered Data Visualizations")
        
        if st.session_state.viz_datasets:
            selected_uid = st.selectbox(
                "Select dataset to visualize:",
                list(st.session_state.viz_datasets.keys()),
                format_func=lambda x: st.session_state.viz_datasets[x]['name'],
                key="viz_dataset_select"
            )
            if isinstance(selected_uid, dict):
                selected_uid = (
                    selected_uid.get("uid")
                    or selected_uid.get("dataset_uid")
                    or json.dumps(selected_uid)
                )

            if selected_uid:
                viz_data = st.session_state.viz_datasets[selected_uid]
                df = viz_data['df']
                dataset_name = viz_data['name']
                is_sampled = viz_data.get('is_sampled', False)
                total_rows = viz_data.get('total_rows', len(df))
                
                # Show sampling warning exactly like screenshot
                if is_sampled:
                    st.warning(f"""
                    ⚠️ **Working with sampled data**  
                    Showing {len(df):,} rows out of {total_rows:,} total rows ({(len(df)/total_rows*100):.0f}% sample)  
                    Visualizations and statistics are based on this sample
                    """)
                else:
                    st.success(f"Loaded complete dataset: {len(df):,} rows with {len(df.columns)} columns")
                
                st.markdown("---")
                
                # --- AI-POWERED ANALYSIS SECTION (NEW LOGIC) ---
                st.markdown("## 🤖 AI-Powered Analysis")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("AI visualization recommendations")
                
                with col2:
                    # Cache key per dataset + row count
                    cache_key = f"{selected_uid}_{len(df)}"
                    
                    # REGENERATE: clear cache + rerun
                    if st.button("🔄 REGENERATE", key=f"regen_ai_viz_{selected_uid}"):
                        if cache_key in st.session_state.ai_analysis_cache:
                            del st.session_state.ai_analysis_cache[cache_key]
                        # Keep compatibility with rest of app
                        if isinstance(st.session_state.auto_viz_triggered, set):
                            st.session_state.auto_viz_triggered.discard(selected_uid)
                        st.rerun()
                
                analysis = None
                
                if not OPENAI_API_KEY:
                    st.warning("⚠️ Set OPENAI_API_KEY in .env file to enable AI analysis")
                else:
                    cache_key = f"{selected_uid}_{len(df)}"
                    
                    # If we don’t have analysis yet, create it once automatically
                    if cache_key not in st.session_state.ai_analysis_cache:
                        with st.spinner("🧠 AI is analyzing your dataset..."):
                            try:
                                analysis = asyncio.run(
                                    analyze_dataset_with_ai(
                                        df,
                                        dataset_name,
                                        is_sampled,
                                        len(df),
                                        total_rows,
                                    )
                                )
                                if isinstance(analysis, dict) and "error" not in analysis:
                                    st.session_state.ai_analysis_cache[cache_key] = analysis
                                else:
                                    err_msg = (
                                        analysis.get("error")
                                        if isinstance(analysis, dict)
                                        else "Unknown error"
                                    )
                                    st.error(f"AI analysis failed: {err_msg}")
                                    analysis = None
                            except Exception as e:
                                st.error(f"AI analysis failed: {e}")
                                analysis = None
                    else:
                        analysis = st.session_state.ai_analysis_cache[cache_key]
                
                # If still None, allow manual trigger
                if analysis is None and OPENAI_API_KEY:
                    if st.button("🤖 GENERATE AI ANALYSIS", key=f"gen_ai_button_{selected_uid}", type="primary"):
                        with st.spinner("🧠 AI is analyzing your dataset..."):
                            try:
                                analysis = asyncio.run(
                                    analyze_dataset_with_ai(
                                        df,
                                        dataset_name,
                                        is_sampled,
                                        len(df),
                                        total_rows,
                                    )
                                )
                                if isinstance(analysis, dict) and "error" not in analysis:
                                    st.session_state.ai_analysis_cache[cache_key] = analysis
                                else:
                                    err_msg = (
                                        analysis.get("error")
                                        if isinstance(analysis, dict)
                                        else "Unknown error"
                                    )
                                    st.error(f"AI analysis failed: {err_msg}")
                                    analysis = None
                            except Exception as e:
                                st.error(f"Failed to run AI analysis: {e}")
                                analysis = None
                
                # --- SHOW AI RECOMMENDATIONS ---
                if analysis is not None:
                    viz_recs = analysis.get("visualizations", [])
                    if not viz_recs:
                        st.info("AI did not return any visualization recommendations.")
                    else:
                        for idx, viz_rec in enumerate(viz_recs):
                            chart_type = str(viz_rec.get("type", "Unknown")).title()
                            reasoning = viz_rec.get("reasoning", "N/A")
                            insights = viz_rec.get("insights", "N/A")
                            
                            with st.expander(
                                f"📊 **{idx+1}. {chart_type} Chart**",
                                expanded=(idx == 0),
                            ):
                                c1, c2 = st.columns([3, 1])
                                
                                with c1:
                                    st.markdown(f"**Why this chart:** {reasoning}")
                                    st.markdown(f"**Expected insights:** {insights}")
                                    
                                    columns = viz_rec.get("columns", {})
                                    if columns:
                                        cols_str = ", ".join(
                                            [str(v) for v in columns.values() if v]
                                        )
                                        st.caption(f"Using columns: {cols_str}")
                                
                                with c2:
                                    if st.button(
                                        "GENERATE CHART",
                                        key=f"gen_chart_{idx}_{selected_uid}",
                                    ):
                                        with st.spinner("Creating visualization..."):
                                            fig = safe_create_visualization(viz_rec, df, idx)
                                            if fig:
                                                st.plotly_chart(
                                                    fig,
                                                    use_container_width=True,
                                                    key=get_unique_key(
                                                        f"ai_chart_{idx}_{selected_uid}"
                                                    ),
                                                )
                                            else:
                                                st.error(
                                                    "Could not create this visualization from the AI recommendation."
                                                )
                
                st.markdown("---")
                
                # --- AUTO-GENERATED VISUALIZATIONS (UPDATED FOR AI-EXTRACTED DATASETS) ---
                # Heuristic: any dataset name ending with "(AI-extracted)" is from Extract & Visualize
                is_ai_extracted = (
                    isinstance(dataset_name, str)
                    and dataset_name.strip().endswith("(AI-extracted)")
                )
                
                st.markdown("## 📊 Auto-Generated Visualizations")
                
                if is_ai_extracted:
                    # Don’t spam the user with the full grid for AI-extracted tables.
                    st.info(
                        "This dataset was **AI-extracted** from the Maryland corpus. "
                        "We’re focusing on the AI-recommended charts above instead of generating "
                        "a full grid of generic visualizations."
                    )
                else:
                    with st.spinner("Creating visualizations..."):
                        figures = create_all_visualizations(df, dataset_name)
                        
                    if figures:
                        # Display in grid layout (2 per row)
                        for i in range(0, len(figures), 2):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if i < len(figures):
                                    st.plotly_chart(
                                        figures[i],
                                        use_container_width=True,
                                        key=get_unique_key(f"auto_viz_left_{i}_{selected_uid}"),
                                    )
                            
                            with col2:
                                if i + 1 < len(figures):
                                    st.plotly_chart(
                                        figures[i + 1],
                                        use_container_width=True,
                                        key=get_unique_key(f"auto_viz_right_{i}_{selected_uid}"),
                                    )
                    else:
                        st.info("No visualizations could be generated")
                
                # Data preview at bottom
                with st.expander("📋 View Raw Data", expanded=False):
                    st.dataframe(df.head(100), use_container_width=True)
        else:
            st.info("📊 No datasets loaded. Please load data from the Search tab.")
    
    # Tab 3: Ask Questions (AI-Powered with RAG)
    with tabs[2]:
        st.header("Ask Questions About Your Data")

        st.info("""
        🔍 **RAG + LLM mode (recommended):**  
        By default, your question is answered using **only the datasets that have been
        ingested and indexed** in the backend.  
        You can turn off RAG to ask general knowledge questions instead.
        """)

        # Question input
        question = st.text_area(
            "Your Question:",
            placeholder=f"As a {selected_persona}, what are the key trends in the ingested datasets related to youth mental health?",
            height=100,
            key="qa_question_input",
        )

        # Answer configuration
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            answer_style = st.selectbox(
                "Answer Style",
                ["Detailed", "Concise", "Technical", "Executive Summary"],
                help="Choose how detailed you want the answer",
                key="answer_style_select",
            )

        with col2:
            include_examples = st.checkbox(
                "Include Examples",
                value=True,
                help="Include practical examples in the answer",
                key="include_examples_check",
            )

        with col3:
            include_references = st.checkbox(
                "Add References",
                value=False,
                help="Include suggested data sources or references",
                key="include_references_check",
            )

        # Optional dataset filter dropdown
        available_datasets = []

        for uid in st.session_state.ingested_datasets:
            name = st.session_state.dataset_names.get(uid, uid)
            available_datasets.append((f"Ingested: {name}", uid))

        for uid, info in st.session_state.viz_datasets.items():
            label = f"Visualized: {info.get('name', uid)}"
            available_datasets.append((label, uid))

        selected_dataset_uids = None
        if available_datasets:
            labels = ["All datasets"] + [lbl for (lbl, _) in available_datasets]
            choice = st.selectbox(
                "Limit answer to a specific dataset (optional):",
                labels,
                index=0,
                help="If you pick a dataset here, RAG will restrict its context to that dataset only.",
            )
            if choice != "All datasets":
                idx = labels.index(choice) - 1
                selected_dataset_uids = [available_datasets[idx][1]]

        # Single toggle that actually controls behavior
        rag_mode = st.checkbox(
            "🔍 Use my ingested/indexed data (RAG mode)",
            value=True,
            help="If ON, your question is answered using the backend /answer endpoint with RAG. If OFF, a general LLM answer is generated without using your data.",
            key="rag_mode_toggle",
        )

        if st.button("🤖 Get AI Answer", type="primary", key="get_ai_answer_btn"):
            if not question:
                st.warning("Please enter a question")
            else:
                # ========= MODE 1: RAG + LLM via backend =========
                if rag_mode:
                    with st.spinner("🧠 Getting RAG + LLM answer..."):
                        rag_data, err = answer_question(
                            question=question,
                            org=None if org == "All" else org,
                            k=8,
                            persona=selected_persona,
                            ingested_only=True,              # only indexed/ingested data
                            dataset_uids=selected_dataset_uids,
                        )

                    if err:
                        st.error(f"Backend error: {err}")
                    elif not rag_data:
                        st.error("No answer returned from backend.")
                    else:
                        answer_text = rag_data.get("answer", "")
                        chunks_used = rag_data.get("chunks_used", 0)
                        llm_enhanced = rag_data.get("llm_enhanced", False)

                        st.success("✅ RAG Answer Generated")
                        st.markdown(
                            f"""
                            <div class='data-card'>
                                <h3>💬 RAG Answer for {selected_persona}</h3>
                                <div style='white-space: pre-wrap;'>{answer_text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.caption(
                            f"Chunks used: {chunks_used} · LLM enhanced: {llm_enhanced}"
                        )

                        st.session_state.chat_history.append(
                            {
                                "question": question,
                                "answer": answer_text,
                                "persona": selected_persona,
                                "style": answer_style,
                                "rag_mode": True,
                                "timestamp": datetime.now(),
                            }
                        )

                # ========= MODE 2: Pure LLM (no RAG) =========
                else:
                    if not OPENAI_API_KEY:
                        st.warning(
                            "⚠️ OpenAI API key not configured in Streamlit. Set OPENAI_API_KEY in your .env file."
                        )
                    else:
                        with st.spinner("🧠 AI is thinking (no data constraints)..."):
                            system_prompt = f"""You are an expert mental health data analyst providing insights for a {selected_persona}.
                            
                            Answer style: {answer_style}
                            Include examples: {include_examples}
                            Include references: {include_references}
                            
                            Tailor your response based on the persona:
                            - Public health researcher: Focus on statistical methods, research design, evidence-based insights
                            - Policy maker: Focus on actionable recommendations, cost-benefit analysis, population impact
                            - Clinician: Focus on clinical applications, patient outcomes, treatment protocols
                            - Epidemiologist: Focus on disease patterns, risk factors, surveillance methods
                            - Data analyst: Focus on technical approaches, data quality, analytical methods
                            """

                            prompt = f"""Question from a {selected_persona}: {question}
                            
                            Please provide a {answer_style.lower()} answer that:
                            1. Directly addresses the question
                            2. Is tailored to the {selected_persona} perspective
                            3. Uses appropriate technical depth
                            {"4. Includes practical examples" if include_examples else ""}
                            {"5. Suggests relevant data sources or references" if include_references else ""}
                            
                            Format your response with clear sections and bullet points where appropriate."""

                            answer, cost = asyncio.run(
                                call_openai(prompt, system_prompt, temperature=0.5)
                            )

                        if answer:
                            st.success("✅ AI Answer Generated (no RAG)")

                            st.markdown(
                                f"""
                                <div class='data-card'>
                                    <h3>💬 AI Answer for {selected_persona}</h3>
                                    <div style='white-space: pre-wrap;'>{answer}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            if cost and cost > 0:
                                st.caption(f"💰 Cost (UI LLM): ${cost:.4f}")

                            st.session_state.chat_history.append(
                                {
                                    "question": question,
                                    "answer": answer,
                                    "persona": selected_persona,
                                    "style": answer_style,
                                    "rag_mode": False,
                                    "timestamp": datetime.now(),
                                }
                            )
                        else:
                            st.error("Failed to generate answer. Please try again.")

        # Chat history display
        if st.session_state.chat_history:
            st.divider()
            st.markdown("### 💬 Chat History")

            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                mode = "RAG" if chat.get("rag_mode") else "LLM"
                with st.expander(
                    f"[{mode}] Q: {chat['question'][:80]}... ({chat['timestamp'].strftime('%H:%M')})",
                    expanded=False,
                ):
                    st.write(
                        f"**Asked at:** {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    st.write(f"**Persona:** {chat['persona']}")
                    st.write(f"**Style:** {chat.get('style', 'Standard')}")
                    st.write("**Answer:**")
                    st.write(chat["answer"])

            if st.button("🗑️ Clear Chat History", key="clear_chat_history_btn"):
                st.session_state.chat_history = []
                st.success("Chat history cleared")
                st.rerun()
  
    # Tab 4: Analytics Dashboard - COMPLETE
    with tabs[3]:
        st.header("Analytics Dashboard")
        
        st.markdown("### 📊 Platform-Wide Analytics")
        
        # Get comprehensive stats
        db_stats = get_database_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{db_stats.get('total_records', 0):,}")
        col2.metric("Unique Datasets", f"{db_stats.get('unique_datasets', 0):,}")
        col3.metric("Data Sources", f"{db_stats.get('data_sources', 0):,}")
        col4.metric("Total Chunks", f"{db_stats.get('total_chunks', 0):,}")
        
        # Source breakdown
        by_source = db_stats.get('by_source', {})
        if by_source:
            st.markdown("### 📁 Records by Source")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig = px.bar(
                    x=list(by_source.keys()),
                    y=list(by_source.values()),
                    title='Records by Organization',
                    labels={'x': 'Organization', 'y': 'Number of Records'},
                    color=list(by_source.values()),
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True, key=get_unique_key("analytics_bar"))
            
            with col2:
                # Pie chart
                fig = px.pie(
                    values=list(by_source.values()),
                    names=list(by_source.keys()),
                    title='Record Distribution',
                    hole=0.3
                )
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True, key=get_unique_key("analytics_pie"))
        
        # Chunks breakdown
        try:
            chunks_stats = get_chunks_stats()
            if chunks_stats['total_chunks'] > 0:
                st.divider()
                st.markdown("### 🧩 Semantic Chunks Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Corpus type breakdown
                    corpus_data = {
                        'Maryland': chunks_stats['maryland_chunks'],
                        'Ingested': chunks_stats['ingested_chunks']
                    }
                    
                    fig = px.pie(
                        values=list(corpus_data.values()),
                        names=list(corpus_data.keys()),
                        title='Chunks by Type',
                        hole=0.3,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True, key=get_unique_key("chunks_type"))
                
                with col2:
                    # By organization
                    by_org = chunks_stats.get('by_org', {})
                    if by_org:
                        fig = px.bar(
                            x=list(by_org.keys()),
                            y=list(by_org.values()),
                            title='Chunks by Organization',
                            color=list(by_org.values()),
                            color_continuous_scale='Plasma'
                        )
                        fig.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig, use_container_width=True, key=get_unique_key("chunks_org"))
        except:
            pass
        
        # Maryland Corpus Analytics
        st.divider()
        st.markdown("### 🏛️ Maryland Corpus Analytics")
        
        try:
            md_stats = get_maryland_corpus_stats()
            
            if md_stats['total_records'] > 0:
                # Overview metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Records", f"{md_stats['total_records']:,}")
                col2.metric("Data Sources", md_stats['unique_sources'])
                col3.metric("Locations", md_stats['unique_locations'])
                
                indexed_stats = get_maryland_corpus_indexed_stats()
                col4.metric("Indexed Chunks", f"{indexed_stats['total_chunks']:,}")
                
                # Source breakdown
                col1, col2 = st.columns(2)
                
                with col1:
                    source_df = get_maryland_corpus_by_source()
                    if not source_df.empty:
                        fig = px.bar(
                            source_df,
                            x='source',
                            y='count',
                            title='Maryland Corpus by Source',
                            color='count',
                            color_continuous_scale='Plasma'
                        )
                        fig.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig, use_container_width=True, key=get_unique_key("md_source"))
                
                with col2:
                    location_df = get_maryland_corpus_by_location()
                    if not location_df.empty:
                        fig = px.bar(
                            location_df.head(10),
                            x='count',
                            y='location',
                            orientation='h',
                            title='Top 10 Locations in Corpus',
                            color='count',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig, use_container_width=True, key=get_unique_key("md_location"))
            else:
                st.info("No Maryland corpus data available yet")
        except Exception as e:
            st.warning(f"Maryland corpus analytics unavailable: {e}")
        
        # Session activity
        st.divider()
        st.markdown("### 📈 Session Activity")
        
        if st.session_state.search_history:
            search_df = pd.DataFrame(st.session_state.search_history)
            
            # Searches over time
            search_df['hour'] = pd.to_datetime(search_df['timestamp']).dt.floor('h')
            searches_by_hour = search_df.groupby('hour').size().reset_index(name='count')
            
            fig = px.line(
                searches_by_hour,
                x='hour',
                y='count',
                title='Search Activity Over Time',
                markers=True
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True, key=get_unique_key("search_timeline"))
            
            # Search types breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                search_type_counts = search_df['search_type'].value_counts()
                fig = px.pie(
                    values=search_type_counts.values,
                    names=search_type_counts.index,
                    title='Search Types Used',
                    hole=0.3
                )
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True, key=get_unique_key("search_types"))
            
            with col2:
                org_counts = search_df['org_filter'].value_counts()
                fig = px.bar(
                    x=org_counts.index,
                    y=org_counts.values,
                    title='Organization Filters Used',
                    labels={'x': 'Organization', 'y': 'Count'},
                    color=org_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True, key=get_unique_key("org_filters"))
        
        # Cost tracking
        st.divider()
        st.markdown("### 💰 Cost Analysis")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total LLM Cost", f"${st.session_state.llm_cost:.4f}")
        col2.metric("Q&A Sessions", len(st.session_state.chat_history))
        col3.metric("Visualizations Generated", len(st.session_state.ai_analysis_cache))
    
    # Tab 5: Export & Share - COMPLETE WITH NAME DISPLAY
    with tabs[4]:
        st.header("💾 Export & Share")
        
        st.markdown("### Export Datasets")
        
        try:
            ingested_df = sql_df("""
                SELECT 
                    dataset_uid,
                    org,
                    dataset_name,
                    row_count,
                    ingested_at
                FROM ingested_datasets
                ORDER BY ingested_at DESC
            """, {})
            
            if not ingested_df.empty:
                # Build display names with fallback
                display_names = []
                for _, row in ingested_df.iterrows():
                    uid = row['dataset_uid']
                    # Check session state first, then database name, then default
                    if uid in st.session_state.dataset_names:
                        name = st.session_state.dataset_names[uid]
                    elif row['dataset_name']:
                        name = row['dataset_name']
                    else:
                        name = f"{row['org']} Dataset {uid}"
                    display_names.append(f"{name} ({row['row_count']} rows)")
                
                # Select dataset to export
                selected_idx = st.selectbox(
                    "Select dataset to export:",
                    range(len(ingested_df)),
                    format_func=lambda x: display_names[x],
                    key="export_dataset_selector"
                )
                
                if selected_idx is not None:
                    dataset_info = ingested_df.iloc[selected_idx]
                    dataset_org = dataset_info['org']
                    dataset_uid = dataset_info['dataset_uid']
                    dataset_name = display_names[selected_idx].split(' (')[0]
                    
                    st.info(f"**Dataset:** {dataset_name}\n**Organization:** {dataset_org}\n**Rows:** {dataset_info['row_count']}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        export_format = st.selectbox("Export Format", ["CSV", "JSON", "Parquet"], key="export_format_select")
                    
                    with col2:
                        max_rows = st.number_input("Max Rows to Export", 1, 50000, 1000, key="export_max_rows")
                    
                    if st.button("📥 Prepare Export", type="primary", key="prepare_export_btn"):
                        with st.spinner("Loading dataset..."):
                            df = preview_dataset(dataset_org, dataset_uid, max_rows)
                            
                            if df is not None and len(df) > 0:
                                # Check if sampled
                                if hasattr(df, 'attrs') and df.attrs.get('is_sampled', False):
                                    st.warning(f"⚠️ Exporting {len(df)} of {df.attrs.get('total_rows', 'unknown')} total rows")
                                
                                # Export based on format
                                if export_format == "CSV":
                                    csv_data = df.to_csv(index=False)
                                    st.download_button(
                                        label="⬇️ Download CSV",
                                        data=csv_data,
                                        file_name=f"{dataset_name.replace(' ', '_')}.csv",
                                        mime="text/csv",
                                        key="download_csv_btn"
                                    )
                                
                                elif export_format == "JSON":
                                    json_data = df.to_json(orient='records', indent=2)
                                    st.download_button(
                                        label="⬇️ Download JSON",
                                        data=json_data,
                                        file_name=f"{dataset_name.replace(' ', '_')}.json",
                                        mime="application/json",
                                        key="download_json_btn"
                                    )
                                
                                elif export_format == "Parquet":
                                    buffer = io.BytesIO()
                                    df.to_parquet(buffer, index=False)
                                    parquet_data = buffer.getvalue()
                                    
                                    st.download_button(
                                        label="⬇️ Download Parquet",
                                        data=parquet_data,
                                        file_name=f"{dataset_name.replace(' ', '_')}.parquet",
                                        mime="application/octet-stream",
                                        key="download_parquet_btn"
                                    )
                                
                                st.success(f"✅ Exported {len(df)} rows")
                                
                                # Preview
                                with st.expander("Preview Export Data"):
                                    st.dataframe(df.head(100))
                            else:
                                st.error("Failed to load dataset for export")
                
                # Generate report
                st.divider()
                st.markdown("### 📄 Generate Summary Report")
                
                if st.button("Generate Report", key="generate_report_btn"):
                    db_stats = get_database_stats()
                    
                    report_content = f"""# Maryland Mental Health Data Platform Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Database Overview
- Total Records: {db_stats.get('total_records', 0):,}
- Unique Datasets: {db_stats.get('unique_datasets', 0):,}
- Data Sources: {db_stats.get('data_sources', 0):,}
- Total Chunks: {db_stats.get('total_chunks', 0):,}

## Records by Source
"""
                    for source, count in db_stats.get('by_source', {}).items():
                        report_content += f"- {source}: {count:,}\n"
                    
                    # Add Maryland corpus stats
                    try:
                        md_stats = get_maryland_corpus_stats()
                        if md_stats['total_records'] > 0:
                            report_content += f"""
## Maryland Corpus
- Total Records: {md_stats['total_records']:,}
- Data Sources: {md_stats['unique_sources']}
- Locations: {md_stats['unique_locations']}
"""
                            indexed_stats = get_maryland_corpus_indexed_stats()
                            report_content += f"- Indexed Chunks: {indexed_stats['total_chunks']:,}\n"
                    except:
                        pass
                    
                    report_content += f"""
## Session Activity
- Datasets Ingested: {len(st.session_state.ingested_datasets)}
- Datasets Indexed: {len(st.session_state.indexed_datasets)}
- Search Queries: {len(st.session_state.search_history)}
- Q&A Sessions: {len(st.session_state.chat_history)}
- LLM Cost: ${st.session_state.llm_cost:.4f}

## Recent Searches
"""
                    for search in st.session_state.search_history[-5:]:
                        report_content += f"- {search['timestamp'].strftime('%Y-%m-%d %H:%M')}: {search['query'][:100]}...\n"
                    
                    st.download_button(
                        label="⬇️ Download Report (Markdown)",
                        data=report_content,
                        file_name=f"platform_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        key="download_report_btn"
                    )
                    
                    with st.expander("Preview Report"):
                        st.markdown(report_content)
            else:
                st.info("No datasets have been ingested yet. Ingest datasets from the Search tab first.")
        
        except Exception as e:
            st.error(f"Error loading export options: {e}")
            st.session_state.last_error = str(e)
    
    # Tab 6: Manage Data - COMPLETE
    with tabs[5]:
        st.header("⚙️ Manage Data")
        
        st.markdown("### 🗃️ Ingested Datasets")
        
        try:
            # Show ingested datasets
            ingested_df = sql_df("""
                SELECT 
                    dataset_uid,
                    org,
                    dataset_name,
                    row_count,
                    ingested_at
                FROM ingested_datasets
                ORDER BY ingested_at DESC
                """, {})
            
            if not ingested_df.empty:
                st.dataframe(ingested_df, use_container_width=True)

                # Delete a dataset
                st.markdown("### 🗑️ Remove Dataset")

                del_uid = st.selectbox(
                    "Select dataset to delete:",
                    ingested_df['dataset_uid'].tolist(),
                    format_func=lambda x: f"{x} — {st.session_state.dataset_names.get(x, '')}",
                    key="delete_dataset_uid"
                )

                if st.button("Delete Selected Dataset", type="secondary"):
                    st.warning("Deletion is permanent. Backend deletion not implemented yet.")
                    # (Hook here for backend delete)
            else:
                st.info("No datasets ingested yet.")
        
        except Exception as e:
            st.error(f"Could not load ingested datasets: {e}")

        st.markdown("---")
        st.markdown("### 🧹 Reset Session State")

        if st.button("Reset All Session State", type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Session state cleared. Reloading…")
            st.rerun()

        st.markdown("---")
        st.caption("Maryland Mental Health Data Platform — Production Version (ALFA8 MindCube)")
        

# ===========================
# Run App
# ===========================
if __name__ == "__main__":
    main()
