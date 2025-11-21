# backend/app/answer.py

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
import time
import json

from .db import engine
from .llm import chat_completion
from . import socrata
from .semantic import embed_text
from .rag import search_dataset_chunks, index_dataset_content

router = APIRouter(prefix="/answer", tags=["RAG"])

# ---- persona prompts ----
PERSONA_SYSTEM_PROMPTS = {
    "policy maker": (
        "You are a concise public-policy briefing assistant. "
        "Use plain language, emphasize implications, trends, and actions. "
        "Prefer bullets, short paragraphs, and a 1-2 sentence bottom line."
    ),
    "clinician": (
        "You are a clinician-facing assistant. Be medically precise; "
        "call out data limitations, risks, and red flags in a neutral tone."
    ),
    "researcher": (
        "You are a research assistant. Highlight methodology, data quality, uncertainty, "
        "and gaps; use technical language when appropriate."
    ),
    "public health researcher": (
        "You are a public health research assistant. Focus on population health impacts, "
        "epidemiological patterns, and evidence-based interventions. "
        "Cite specific statistics and data points when available."
    ),
}

def persona_prompt(p: str | None) -> str:
    if not p:
        return ("You are a helpful assistant for public-health analytics. "
                "Be clear, cautious, and note data limitations.")
    key = (p or "").strip().lower()
    return PERSONA_SYSTEM_PROMPTS.get(
        key,
        "You are a helpful assistant for public-health analytics. Be clear and note limitations."
    )

class AnswerQuery(BaseModel):
    question: str
    org: str = "CDC"
    k: int = 5
    persona: str | None = None
    use_actual_data: bool = True  # Default to using actual data when available

class IndexDatasetRequest(BaseModel):
    org: str
    uid: str
    limit_rows: int = 5000

def _vec_literal(vec: list[float]) -> str:
    """Convert Python list -> SQL vector literal for pgvector."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

def _build_data_prompt(question: str, chunks: list[dict]) -> str:
    """Build a prompt from actual data chunks"""
    lines = []
    lines.append("You are analyzing actual public health data to answer questions.")
    lines.append("Use the following data excerpts to provide a factual, data-driven answer.")
    lines.append("")
    lines.append("=== Actual Data from Datasets ===")
    
    for i, chunk in enumerate(chunks, 1):
        dataset_name = chunk.get("dataset_name", "Unknown Dataset")
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        
        lines.append(f"\n[{i}] From: {dataset_name}")
        if metadata.get("group_by"):
            lines.append(f"    Filtered by: {metadata['group_by']}")
        lines.append(f"    Data:\n{content[:2000]}\n")  # Limit content length
    
    lines.append("=== Your Task ===")
    lines.append("Answer the following question using the actual data provided above.")
    lines.append("Be specific and cite actual numbers, trends, and patterns from the data.")
    lines.append("If the data doesn't contain enough information, say so clearly.")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Format your answer with:")
    lines.append("- A clear 1-2 sentence summary")
    lines.append("- 3-5 specific data points with actual numbers from the data")
    lines.append("- Any important caveats or limitations")
    
    return "\n".join(lines)

def _build_metadata_prompt(question: str, rows: list[dict]) -> str:
    """Build a prompt from dataset metadata (fallback)"""
    if not rows:
        return f"No datasets found. Please answer based on general knowledge: {question}"
        
    lines = []
    lines.append("You are given context snippets from CDC/SAMHSA dataset descriptions.")
    lines.append("Note: This is metadata only, not actual data. Answer based on what these datasets claim to contain.")
    lines.append("\n=== Dataset Descriptions ===")
    
    for i, r in enumerate(rows, 1):
        uid = r.get("uid", "")
        name = r.get("name", "") or ""
        desc = (r.get("description", "") or "").strip()[:1200]
        link = f"https://data.{r.get('org', 'cdc').lower()}.gov/d/{uid}" if uid else ""
        lines.append(f"\n[{i}] {name} (uid: {uid})")
        if link:
            lines.append(f"    Link: {link}")
        if desc:
            lines.append(f"    Description: {desc}")
    
    lines.append("\n=== Task ===")
    lines.append("Answer the question based on what these dataset descriptions indicate.")
    lines.append("Be clear that you're answering based on metadata, not actual data analysis.")
    lines.append(f"\nQuestion: {question}")
    
    return "\n".join(lines)

@router.post("/")
async def generate_answer(q: AnswerQuery):
    """
    Enhanced RAG that prioritizes actual data chunks over metadata
    """
    print(f"[Answer] Received question: {q.question[:100]}... with persona: {q.persona}")
    start_time = time.time()
    
    # Strategy 1: Try to use actual data chunks if available
    if q.use_actual_data:
        try:
            print("[Answer] Searching for actual data chunks...")
            data_chunks = await search_dataset_chunks(q.question, q.org, q.k)
            
            if data_chunks:
                print(f"[Answer] Found {len(data_chunks)} relevant data chunks")
                
                # Build prompt with actual data
                user_prompt = _build_data_prompt(q.question, data_chunks)
                system_prompt = persona_prompt(q.persona)
                
                # Generate answer from actual data
                answer_text = await chat_completion(
                    user_prompt,
                    system=system_prompt,
                    temperature=0.2,
                    max_tokens=700
                )
                
                # Prepare sources from actual datasets
                sources = []
                seen_uids = set()
                for chunk in data_chunks:
                    uid = chunk.get("dataset_uid")
                    if uid and uid not in seen_uids:
                        seen_uids.add(uid)
                        sources.append({
                            "uid": uid,
                            "name": chunk.get("dataset_name", f"Dataset {uid}"),
                            "description": f"Data chunk: {chunk.get('summary', '')[:200]}...",
                            "link": f"https://data.{q.org.lower()}.gov/d/{uid}"
                        })
                
                elapsed = time.time() - start_time
                print(f"[Answer] Success using actual data in {elapsed:.2f}s")
                
                return {
                    "ok": True,
                    "question": q.question,
                    "persona": q.persona,
                    "answer": answer_text,
                    "sources": sources,
                    "mode": "actual_data",
                    "chunks_used": len(data_chunks)
                }
                
        except Exception as e:
            print(f"[Answer] Failed to use data chunks: {e}")
    
    # Strategy 2: Use semantic search on metadata
    try:
        print("[Answer] Falling back to semantic search on metadata...")
        
        # Check if semantic_index exists and has data
        with engine.begin() as conn:
            check = conn.execute(
                text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'semantic_index'
                    )
                """)
            ).scalar()
            
            if check:
                count = conn.execute(
                    text("SELECT COUNT(*) FROM semantic_index WHERE org = :o"),
                    {"o": q.org}
                ).scalar()
                
                if count > 0:
                    # Try semantic search
                    query_text = f"{q.persona}: {q.question}" if q.persona else q.question
                    emb = await embed_text(query_text)
                    vec = _vec_literal(emb)
                    
                    rows = conn.execute(
                        text("""
                            SELECT uid, name, description
                            FROM semantic_index
                            WHERE org = :o
                            ORDER BY embedding <-> CAST(:v AS vector)
                            LIMIT :k
                        """),
                        {"o": q.org, "v": vec, "k": q.k}
                    ).mappings().all()
                    
                    if rows:
                        print(f"[Answer] Found {len(rows)} semantic matches")
                        user = _build_metadata_prompt(q.question, [dict(r) for r in rows])
                        system = persona_prompt(q.persona)
                        
                        text_out = await chat_completion(
                            user,
                            system=system,
                            temperature=0.2,
                            max_tokens=500
                        )
                        
                        sources = []
                        for r in rows:
                            obj = dict(r)
                            uid = obj.get("uid")
                            obj["link"] = f"https://data.{q.org.lower()}.gov/d/{uid}" if uid else None
                            sources.append(obj)
                        
                        elapsed = time.time() - start_time
                        print(f"[Answer] Success via semantic metadata in {elapsed:.2f}s")
                        
                        return {
                            "ok": True,
                            "question": q.question,
                            "persona": q.persona,
                            "answer": text_out,
                            "sources": sources,
                            "mode": "metadata_semantic",
                        }
                        
    except Exception as e:
        print(f"[Answer] Semantic search failed: {e}")
    
    # Strategy 3: Keyword search fallback
    try:
        print("[Answer] Falling back to keyword search...")
        hits = await socrata.search_catalog(q.org, q.question, limit=q.k)
        
        if hits:
            print(f"[Answer] Found {len(hits)} keyword matches")
            user = _build_metadata_prompt(q.question, hits)
            system = persona_prompt(q.persona)
            
            text_out = await chat_completion(
                user,
                system=system,
                temperature=0.2,
                max_tokens=500
            )
            
            return {
                "ok": True,
                "question": q.question,
                "persona": q.persona,
                "answer": text_out,
                "sources": hits,
                "mode": "keyword_fallback"
            }
            
    except Exception as e:
        print(f"[Answer] Keyword search failed: {e}")
    
    # No data found
    return {
        "ok": True,
        "question": q.question,
        "persona": q.persona,
        "answer": (
            "I couldn't find relevant data to answer your question. This could be because:\n\n"
            "• No datasets have been ingested and indexed yet\n"
            "• The question doesn't match any available data\n"
            "• The search services are temporarily unavailable\n\n"
            "**Suggestions:**\n"
            "1. First, search and ingest relevant datasets using the 'Search & Results' tab\n"
            "2. Click 'Index for RAG' on ingested datasets to make them searchable\n"
            "3. Try rephrasing your question or using broader terms"
        ),
        "sources": [],
        "mode": "no_data",
    }

@router.post("/index_dataset")
async def index_dataset_endpoint(req: IndexDatasetRequest):
    """
    Index a specific dataset's actual content for RAG queries.
    This loads the parquet file and creates searchable chunks.
    """
    try:
        result = await index_dataset_content(
            org=req.org,
            uid=req.uid,
            limit_rows=req.limit_rows
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/indexed_status")
async def get_indexed_status(org: str = "CDC"):
    """
    Check the status of indexed datasets
    """
    with engine.begin() as conn:
        # Check data chunks
        data_chunks_exists = conn.execute(
            text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'data_chunks'
                )
            """)
        ).scalar()
        
        data_chunks_count = 0
        indexed_datasets = []
        
        if data_chunks_exists:
            # Count total chunks
            data_chunks_count = conn.execute(
                text("SELECT COUNT(*) FROM data_chunks WHERE org = :o"),
                {"o": org}
            ).scalar()
            
            # Get indexed datasets
            rows = conn.execute(
                text("""
                    SELECT DISTINCT 
                        dataset_uid,
                        COUNT(*) as chunk_count,
                        MAX(created_at) as last_indexed
                    FROM data_chunks
                    WHERE org = :o
                    GROUP BY dataset_uid
                    ORDER BY last_indexed DESC
                    LIMIT 10
                """),
                {"o": org}
            ).mappings().all()
            indexed_datasets = [dict(r) for r in rows]
        
        # Check semantic index
        semantic_count = 0
        semantic_exists = conn.execute(
            text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'semantic_index'
                )
            """)
        ).scalar()
        
        if semantic_exists:
            semantic_count = conn.execute(
                text("SELECT COUNT(*) FROM semantic_index WHERE org = :o"),
                {"o": org}
            ).scalar()
        
        return {
            "org": org,
            "data_chunks": {
                "exists": data_chunks_exists,
                "total_chunks": data_chunks_count,
                "indexed_datasets": indexed_datasets
            },
            "semantic_index": {
                "exists": semantic_exists,
                "count": semantic_count
            },
            "ready": data_chunks_count > 0 or semantic_count > 0
        }