"""
FastAPI Backend for Maryland Mental Health Data Platform
Complete version with all fixes for semantic search, visualization, and export
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import re
import httpx

# Fixed imports for Docker - use app. prefix
from app.ingest import (
    ingest_dataset,
    get_dataset_from_minio,
    ensure_bucket_exists,
    fetch_dataset_from_socrata,
)
from app.rag import index_dataset_for_rag, search_chunks
from app.socrata import search_catalog, fetch_rows
from app.db_utils import sql_df

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Maryland Mental Health Data Platform API",
    version="4.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# Pydantic Models
# ===========================


class IngestRequest(BaseModel):
    org: str
    dataset_uid: Optional[str] = None
    pick_uid: Optional[str] = None
    auto_index: bool = False


class IndexRequest(BaseModel):
    org: str
    uid: Optional[str] = None
    dataset_uid: Optional[str] = None
    limit_rows: int = 5000


class SemanticQuery(BaseModel):
    story: str
    org: Optional[str] = None
    k: int = 10
    persona: Optional[str] = None
    filter_ingested_only: bool = False


class AnswerRequest(BaseModel):
    question: str
    org: Optional[str] = None
    k: int = 5
    use_actual_data: bool = True
    persona: Optional[str] = None
    ingested_only: bool = True
    dataset_uids: Optional[List[str]] = None  # ⬅ NEW: optional filter


# ===========================
# Health & Status Endpoints
# ===========================


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Maryland Mental Health Data Platform",
    }


@app.get("/database_stats")
async def get_database_statistics():
    """Get real database statistics for the overview panel"""
    try:
        # Get total ROWS from ingested datasets (not total datasets)
        records_df = sql_df(
            """
            SELECT 
                COALESCE(SUM(row_count), 0) as total_records
            FROM ingested_datasets
            """,
            params=None,
        )

        # Get unique datasets and sources
        datasets_df = sql_df(
            """
            SELECT 
                COUNT(DISTINCT dataset_uid) as unique_datasets,
                COUNT(DISTINCT org) as data_sources
            FROM ingested_datasets
            """,
            params=None,
        )

        # Get chunks breakdown
        chunks_df = sql_df(
            """
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(DISTINCT dataset_uid) as unique_chunk_datasets
            FROM chunks
            """,
            params=None,
        )

        # Get corpus breakdown
        corpus_df = sql_df(
            """
            SELECT 
                CASE 
                    WHEN chunk_id LIKE 'MD_%' OR dataset_uid LIKE 'maryland_data_%' 
                    THEN 'maryland'
                    ELSE 'ingested'
                END as corpus_type,
                COUNT(*) as count
            FROM chunks
            GROUP BY corpus_type
            """,
            params=None,
        )

        # Get locations
        try:
            locations_df = sql_df(
                """
                SELECT COUNT(DISTINCT location) as unique_locations
                FROM datasets
                WHERE location IS NOT NULL AND location != ''
                """,
                params=None,
            )
            unique_locations = (
                int(locations_df.iloc[0]["unique_locations"])
                if not locations_df.empty
                else 0
            )
        except Exception:
            unique_locations = 0

        stats = {
            "total_records": int(records_df.iloc[0]["total_records"])
            if not records_df.empty
            else 0,
            "unique_datasets": int(datasets_df.iloc[0]["unique_datasets"])
            if not datasets_df.empty
            else 0,
            "data_sources": int(datasets_df.iloc[0]["data_sources"])
            if not datasets_df.empty
            else 0,
            "total_chunks": int(chunks_df.iloc[0]["total_chunks"])
            if not chunks_df.empty
            else 0,
            "unique_locations": unique_locations,
            "corpus_breakdown": {
                "maryland": 0,
                "ingested": 0,
            },
        }

        # Add corpus breakdown
        if not corpus_df.empty:
            for _, row in corpus_df.iterrows():
                corpus_type = row["corpus_type"]
                if corpus_type in stats["corpus_breakdown"]:
                    stats["corpus_breakdown"][corpus_type] = int(row["count"])

        return stats

    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {
            "total_records": 0,
            "unique_datasets": 0,
            "data_sources": 0,
            "total_chunks": 0,
            "unique_locations": 0,
            "corpus_breakdown": {"maryland": 0, "ingested": 0},
        }


@app.get("/rag_status")
async def rag_status(org: Optional[str] = Query(None)):
    """Get RAG indexing status"""
    try:
        # Get indexed datasets count
        if org:
            df = sql_df(
                """
                SELECT 
                    COUNT(DISTINCT dataset_uid) as indexed_datasets,
                    COUNT(*) as total_chunks
                FROM chunks
                WHERE org = %s
                """,
                params=(org,),
            )
        else:
            df = sql_df(
                """
                SELECT 
                    COUNT(DISTINCT dataset_uid) as indexed_datasets,
                    COUNT(*) as total_chunks
                FROM chunks
                """,
                params=None,
            )

        indexed_datasets = (
            int(df.iloc[0]["indexed_datasets"]) if not df.empty else 0
        )
        total_chunks = int(df.iloc[0]["total_chunks"]) if not df.empty else 0

        # Get ingested datasets for RAG context
        ingested_df = sql_df(
            """
            SELECT 
                dataset_uid,
                org,
                dataset_name,
                row_count,
                indexed
            FROM ingested_datasets
            WHERE indexed = TRUE
            """,
            params=None,
        )

        ingested_uids = (
            ingested_df["dataset_uid"].tolist() if not ingested_df.empty else []
        )

        # Get chunks only from ingested datasets
        if ingested_uids:
            placeholders = ",".join(["%s"] * len(ingested_uids))
            ingested_chunks_df = sql_df(
                f"""
                SELECT COUNT(*) as count
                FROM chunks
                WHERE dataset_uid IN ({placeholders})
                """,
                params=tuple(ingested_uids),
            )
            ingested_chunks_count = (
                int(ingested_chunks_df.iloc[0]["count"])
                if not ingested_chunks_df.empty
                else 0
            )
        else:
            ingested_chunks_count = 0

        return {
            "indexed_datasets": indexed_datasets,
            "total_chunks": total_chunks,
            "ingested_chunks": ingested_chunks_count,
            "ingested_dataset_uids": ingested_uids,
        }

    except Exception as e:
        logger.error(f"Failed to get RAG status: {e}")
        return {
            "indexed_datasets": 0,
            "total_chunks": 0,
            "ingested_chunks": 0,
            "ingested_dataset_uids": [],
        }


# ===========================
# Corpus Text Endpoint (for Extract & Visualize)
# ===========================


@app.get("/corpus/text")
async def get_corpus_text_endpoint(
    uid: str = Query(..., description="Internal dataset_uid from chunks table"),
    max_chunks: int = Query(25, le=100, description="Max number of chunks"),
    max_chars: int = Query(8000, le=20000, description="Max characters of text"),
):
    """
    Return concatenated corpus text for a semantic dataset (Maryland corpus, etc.),
    so the UI can run an LLM to extract a tabular dataset for visualization.
    """
    try:
        logger.info(f"Fetching corpus text for uid={uid}, max_chunks={max_chunks}")

        df = sql_df(
            """
            SELECT content
            FROM chunks
            WHERE dataset_uid = %s
            LIMIT %s
            """,
            params=(uid, max_chunks),
        )

        if df.empty:
            raise HTTPException(
                status_code=404, detail="No chunks found for this dataset_uid"
            )

        text_parts = df["content"].astype(str).tolist()
        full_text = "\n\n".join(text_parts)

        if len(full_text) > max_chars:
            full_text = full_text[:max_chars]

        return {
            "uid": uid,
            "text": full_text,
            "chunks_used": len(text_parts),
            "truncated": len(full_text) >= max_chars,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_corpus_text error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# Keyword Search Endpoint
# ===========================


@app.get("/catalog/search")
async def catalog_search(
    q: str = Query(..., description="Search query"),
    org: str = Query("CDC", description="Organization or 'All'"),
    limit: int = Query(10, le=50, description="Max results"),
):
    """Search external data catalogs for datasets using keyword search"""
    try:
        logger.info(f"Catalog search: query='{q}', org={org}, limit={limit}")

        # Domain mapping
        domain_map = {
            "CDC": "data.cdc.gov",
            "SAMHSA": "data.samhsa.gov",
            "Maryland": "data.maryland.gov",
        }

        # Handle "All" - search all sources
        if org == "All":
            all_results: List[Dict[str, Any]] = []
            for org_name in ["CDC", "SAMHSA", "Maryland"]:
                domain = domain_map[org_name]
                try:
                    results = await _search_single_catalog(q, domain, org_name, limit)
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(f"Failed to search {org_name}: {e}")

            # Limit to requested amount
            all_results = all_results[:limit]
            return {"results": all_results}

        # Single org search
        domain = domain_map.get(org, "data.cdc.gov")
        results = await _search_single_catalog(q, domain, org, limit)
        return {"results": results}

    except Exception as e:
        logger.error(f"Catalog search error: {e}", exc_info=True)
        return {"results": [], "error": str(e)}


async def _search_single_catalog(
    query: str, domain: str, org: str, limit: int
) -> List[Dict]:
    """Helper to search a single Socrata catalog"""
    try:
        url = "https://api.us.socrata.com/api/catalog/v1"
        params = {
            "q": query,
            "domains": domain,
            "only": "datasets",
            "limit": limit,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, params=params)

            if response.status_code != 200:
                logger.error(f"Socrata API returned {response.status_code} for {domain}")
                return []

            data = response.json()
            results: List[Dict[str, Any]] = []

            for item in data.get("results", []):
                resource = item.get("resource", {})
                dataset_id = resource.get("id", "")

                if not dataset_id:
                    continue

                results.append(
                    {
                        "uid": dataset_id,  # Actual dataset ID to use for ingestion
                        "dataset_uid": dataset_id,  # explicit for UI
                        "name": resource.get("name", "Unknown Dataset"),
                        "title": resource.get("name", "Unknown Dataset"),
                        "description": resource.get(
                            "description", "No description available"
                        ),
                        "org": org,
                        "link": item.get("link") or f"https://{domain}/d/{dataset_id}",
                        "source": "external_api",
                        "corpus_type": "external",
                        "can_ingest": True,
                    }
                )

            logger.info(f"Found {len(results)} results from {domain}")
            return results

    except Exception as e:
        logger.error(f"Error searching {domain}: {e}")
        return []


# ===========================
# Semantic Search Endpoint
# ===========================


@app.post("/semantic/search")
async def semantic_search(query: SemanticQuery):
    """Semantic search using pgvector with proper corpus detection"""
    try:
        if not query.story or query.story.strip() == "":
            return {
                "used_semantic": False,
                "results": [],
                "message": "Please provide a search query",
                "total_chunks_searched": 0,
            }

        logger.info(f"Semantic search: query='{query.story[:100]}', org={query.org}")

        # Define corpus_types based on filter
        corpus_types = None
        if query.filter_ingested_only:
            corpus_types = ["ingested"]

        # Search chunks
        chunks = search_chunks(
            query.story,
            org=query.org,
            limit=query.k,
            corpus_types=corpus_types,
        )

        if not chunks:
            logger.info("No semantic matches found")
            return {
                "used_semantic": True,
                "results": [],
                "message": "No semantic matches found for this query.",
                "total_chunks_searched": 0,
            }

        # Transform chunks to dataset results
        datasets_map: Dict[str, Dict[str, Any]] = {}

        for chunk in chunks:
            uid = chunk.get("dataset_uid")
            content = chunk.get("content", "")
            corpus_type = chunk.get("corpus_type", "unknown")
            org_value = chunk.get("org", "Unknown")
            metadata = chunk.get("metadata", {})

            if not uid:
                # Skip weird rows with no uid
                continue

            if uid not in datasets_map:
                # Properly detect Maryland corpus
                is_maryland_corpus = (
                    corpus_type == "maryland"
                    or uid.startswith("maryland_data_")
                    or (corpus_type == "unknown" and "maryland_data_" in uid)
                )

                if is_maryland_corpus:
                    # Maryland pre-indexed corpus dataset
                    if "maryland_data_" in uid:
                        parts = uid.split("_")
                        if len(parts) >= 3:
                            actual_dataset_id = parts[-1]

                            # PubMed vs Maryland Open Data
                            if "pubmed" in uid.lower() or "pmid" in str(metadata).lower():
                                display_name = (
                                    f"PubMed Research Study (PMID: {actual_dataset_id})"
                                )
                                source_url = (
                                    f"https://pubmed.ncbi.nlm.nih.gov/{actual_dataset_id}/"
                                )
                                actual_org = "PubMed"
                            else:
                                display_name = (
                                    f"Maryland Health Dataset #{actual_dataset_id}"
                                )
                                source_url = (
                                    f"https://data.maryland.gov/d/{actual_dataset_id}"
                                )
                                actual_org = "Maryland"
                        else:
                            actual_dataset_id = uid
                            display_name = f"Maryland Corpus Dataset {uid}"
                            source_url = ""
                            actual_org = "Maryland"
                    else:
                        actual_dataset_id = uid
                        display_name = f"Maryland Dataset {uid}"
                        source_url = f"https://data.maryland.gov/d/{uid}"
                        actual_org = "Maryland"

                    can_ingest = False  # corpus-only, not direct Socrata ids

                    # Try to get better title from metadata
                    if isinstance(metadata, dict):
                        title = metadata.get("title", "") or metadata.get("name", "")
                        if title and title != "None" and len(title) > 5:
                            display_name = title

                else:
                    # Regular ingested dataset that can be (re)ingested / previewed
                    actual_dataset_id = uid

                    # Determine org
                    if org_value and org_value != "Unknown":
                        actual_org = org_value
                    elif query.org and query.org != "All":
                        actual_org = query.org
                    else:
                        # heuristic fallback
                        low = (content[:200] or "").lower()
                        if "samhsa" in low:
                            actual_org = "SAMHSA"
                        elif "maryland" in low:
                            actual_org = "Maryland"
                        else:
                            actual_org = "CDC"

                    # Extract title from metadata if available
                    display_name = None
                    if isinstance(metadata, dict):
                        display_name = metadata.get("title") or metadata.get("name")

                    if not display_name:
                        display_name = f"{actual_org} Dataset {uid}"

                    # Build source URL based on org
                    if actual_org == "CDC":
                        source_url = f"https://data.cdc.gov/d/{actual_dataset_id}"
                    elif actual_org == "SAMHSA":
                        source_url = f"https://data.samhsa.gov/d/{actual_dataset_id}"
                    elif actual_org == "Maryland":
                        source_url = f"https://data.maryland.gov/d/{actual_dataset_id}"
                    else:
                        source_url = ""

                    can_ingest = True

                # Extract meaningful description
                description = (
                    content[:500] if content else "No description available"
                )
                if content and len(content) > 50:
                    sentences = content.split(".")
                    if sentences and 20 < len(sentences[0]) < 300:
                        description = sentences[0].strip() + "."

                datasets_map[uid] = {
                    "uid": actual_dataset_id,          # cleaned usable id
                    "dataset_uid": actual_dataset_id,  # explicit for UI
                    "internal_uid": uid,               # internal corpus uid
                    "name": display_name,
                    "title": display_name,
                    "org": actual_org,
                    "link": source_url,
                    "description": description,
                    "similarity": chunk.get("similarity", 0.0),
                    "score": chunk.get("similarity", 0.0),
                    "source": "semantic",
                    "corpus_type": corpus_type,
                    "chunk_count": 1,
                    "can_ingest": can_ingest,
                }
            else:
                datasets_map[uid]["chunk_count"] += 1
                if chunk.get("similarity", 0) > datasets_map[uid]["similarity"]:
                    datasets_map[uid]["similarity"] = chunk["similarity"]
                    datasets_map[uid]["score"] = chunk["similarity"]

        # Convert to list and sort
        results = list(datasets_map.values())
        results.sort(key=lambda x: x["similarity"], reverse=True)

        for i, result in enumerate(results):
            result["rank"] = i + 1

        logger.info(
            "Semantic search returned %d datasets - Maryland corpus: %d, Ingestable: %d",
            len(results),
            sum(1 for r in results if not r["can_ingest"]),
            sum(1 for r in results if r["can_ingest"]),
        )

        return {
            "used_semantic": True,
            "results": results,
            "model": "text-embedding-3-small",
            "total_chunks": len(chunks),
            "filtered_to_ingested": query.filter_ingested_only,
        }

    except Exception as e:
        logger.error(f"Semantic search error: {e}", exc_info=True)
        return {
            "used_semantic": False,
            "results": [],
            "error": str(e),
            "message": f"Search failed: {str(e)}",
        }


# ===========================
# Ingestion Endpoints
# ===========================


@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest):
    """Ingest a dataset from Socrata to MinIO and PostgreSQL"""

    dataset_uid = request.dataset_uid or request.pick_uid

    if not dataset_uid:
        raise HTTPException(
            status_code=400, detail="dataset_uid or pick_uid is required"
        )

    # Clean the dataset UID if it's from Maryland corpus
    if "maryland_data_" in dataset_uid:
        # Extract clean ID from maryland_data_source_id format
        parts = dataset_uid.split("_")
        if len(parts) >= 4:
            dataset_uid = parts[-1]

    try:
        ensure_bucket_exists()

        # Log the ingestion attempt
        logger.info(f"Attempting to ingest: org={request.org}, uid={dataset_uid}")

        # Perform ingestion
        result = ingest_dataset(request.org, dataset_uid, request.auto_index)

        if not result.get("success"):
            # Try with different org if it fails
            if request.org == "Maryland":
                # Try Maryland Open Data domain
                logger.info("Retrying with Maryland Open Data domain")
                from app.ingest import DOMAINS

                DOMAINS["Maryland"] = "data.maryland.gov"
                result = ingest_dataset(request.org, dataset_uid, request.auto_index)

            if not result.get("success"):
                raise HTTPException(
                    status_code=400,
                    detail=result.get("reason", "Ingestion failed"),
                )

        # Update the dataset name in the database if needed
        if result.get("success"):
            try:
                sql_df(
                    """
                    UPDATE ingested_datasets 
                    SET dataset_name = %s 
                    WHERE dataset_uid = %s AND org = %s
                    """,
                    params=(
                        f"{request.org} Dataset {dataset_uid}",
                        dataset_uid,
                        request.org,
                    ),
                )
            except Exception:
                pass

        return result

    except Exception as e:
        logger.error(f"Ingestion endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index_dataset")
async def index_dataset_endpoint(request: IndexRequest):
    """Index a dataset for RAG"""

    dataset_uid = request.uid or request.dataset_uid

    if not dataset_uid:
        raise HTTPException(status_code=400, detail="uid or dataset_uid is required")

    try:
        result = index_dataset_for_rag(request.org, dataset_uid)

        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Indexing failed"),
            )

        return result

    except Exception as e:
        logger.error(f"Indexing endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# RAG Q&A Endpoints
# ===========================

@app.post("/answer")
async def answer_question(request: AnswerRequest):
    """Answer questions using RAG (optionally limited to specific dataset_uids)"""
    try:
        dataset_filter = request.dataset_uids or []
        logger.info(
            f"Answering question: '{request.question[:100]}', "
            f"ingested_only={request.ingested_only}, "
            f"dataset_filter={dataset_filter}"
        )

        # Get list of ingested + indexed dataset UIDs if filtering
        ingested_uids: List[str] = []
        if request.ingested_only:
                    # Optional: limit RAG to specific dataset(s)
            allowed_datasets = set(request.dataset_uids) if request.dataset_uids else None

            ingested_df = sql_df(
                """
                SELECT dataset_uid
                FROM ingested_datasets
                WHERE indexed = TRUE
                """,
                params=None,
            )

            if not ingested_df.empty:
                ingested_uids = ingested_df["dataset_uid"].tolist()
                logger.info(
                    f"RAG will search only {len(ingested_uids)} ingested/indexed datasets"
                )
            else:
                logger.warning("No ingested datasets found for RAG")
                return {
                    "answer": "No datasets have been ingested and indexed yet. Please ingest some datasets first.",
                    "sources": [],
                    "chunks_used": 0,
                    "searched_ingested_only": True,
                    "dataset_filter_used": [],
                }

            # If user asked for specific dataset_uids, make sure they are ingested/indexed
            if dataset_filter:
                available = [uid for uid in dataset_filter if uid in ingested_uids]
                missing = [uid for uid in dataset_filter if uid not in ingested_uids]

                if missing and not available:
                    # None of the requested datasets are indexed
                    msg = (
                        "The selected dataset(s) are not ingested and indexed yet. "
                        "Please ingest and index them first from the Search tab."
                    )
                    logger.warning(msg + f" Missing: {missing}")
                    return {
                        "answer": msg,
                        "sources": [],
                        "chunks_used": 0,
                        "searched_ingested_only": True,
                        "dataset_filter_used": dataset_filter,
                    }

                # Restrict filter to the ones we actually have indexed
                if missing:
                    logger.warning(
                        f"Some selected dataset_uids are not indexed and will be ignored: {missing}"
                    )
                dataset_filter = available

        # Semantic search for relevant chunks
        chunks = search_chunks(
            request.question,
            org=request.org,
            limit=request.k,
            corpus_types=["ingested"] if request.ingested_only else None,
        )
                # If a dataset filter is set, keep only chunks from those datasets
        if allowed_datasets:
            chunks = [c for c in chunks if c.get("dataset_uid") in allowed_datasets]

        # Apply dataset filter on chunks (if provided)
        if dataset_filter:
            chunks = [
                c for c in chunks if c.get("dataset_uid") in dataset_filter
            ]

        if not chunks:
            msg = "I couldn't find relevant information in the indexed datasets."
            if allowed_datasets:
                msg = (
                    "The selected dataset(s) are not ingested and indexed yet. "
                    "Please ingest and index them first from the Search tab."
                )
            return {
                "answer": msg,
                "sources": [],
                "chunks_used": 0,
                "searched_ingested_only": request.ingested_only
            }


        # Build context from chunks
        context_chunks = chunks[: request.k]
        context = "\n\n".join(
            [f"[Source {i+1}] {chunk['content'][:500]}" for i, chunk in enumerate(context_chunks)]
        )

        # Get source datasets
        sources = []
        seen_uids = set()
        for chunk in context_chunks:
            uid = chunk.get("dataset_uid")
            if uid and uid not in seen_uids:
                seen_uids.add(uid)
                sources.append(
                    {
                        "name": f"Dataset: {uid}",
                        "uid": uid,
                        "similarity": chunk.get("similarity", 0),
                        "corpus_type": chunk.get("corpus_type", "unknown"),
                    }
                )

        # Try to use OpenAI for enhanced answer
        openai_key = os.getenv("OPENAI_API_KEY")

        if openai_key and request.use_actual_data:
            try:
                import openai

                openai.api_key = openai_key

                persona_context = ""
                if request.persona:
                    persona_context = (
                        f"\nYou are answering for a {request.persona}. "
                        f"Tailor your response accordingly."
                    )

                if dataset_filter:
                    scope_context = (
                        "You are searching ONLY the specified ingested datasets: "
                        + ", ".join(dataset_filter)
                    )
                else:
                    scope_context = (
                        "You are searching ONLY ingested datasets."
                        if request.ingested_only
                        else "You are searching the full corpus."
                    )

                system_prompt = f"""You are an expert mental health data analyst. 
{scope_context}
Provide accurate, evidence-based answers using only the provided context.
Cite specific statistics and findings from the data.{persona_context}"""

                user_prompt = f"""Question: {request.question}

Context from indexed datasets:
{context}

Please provide a comprehensive answer based on the data above."""

                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=1000,
                )

                answer = response.choices[0].message.content
                logger.info("Generated LLM-enhanced RAG answer")

                return {
                    "answer": answer,
                    "sources": sources,
                    "chunks_used": len(chunks),
                    "llm_enhanced": True,
                    "model": "gpt-4o-mini",
                    "searched_ingested_only": request.ingested_only,
                    "dataset_filter_used": dataset_filter,
                }

            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}")

        # Fallback: Context-based response
        answer = f"""Based on the indexed data:

{context[:1500]}

Sources: {len(sources)} datasets with {len(chunks)} relevant segments."""

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(chunks),
            "llm_enhanced": False,
            "searched_ingested_only": request.ingested_only,
            "dataset_filter_used": dataset_filter,
        }

    except Exception as e:
        logger.error(f"Answer question error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



# ===========================
# Dataset Preview and Info Endpoints
# ===========================


@app.get("/datasets/preview")
async def preview_dataset(
    org: str = Query(...),
    uid: str = Query(...),
    rows: int = Query(200, le=50000),
):
    """Preview dataset content with transparent sampling"""
    try:
        logger.info(f"Preview dataset: org={org}, uid={uid}, rows={rows}")

        # Try to get from MinIO first
        df = get_dataset_from_minio(org, uid)

        if df.empty:
            # Fallback: Fetch from Socrata directly
            logger.info("Dataset not in MinIO, fetching from Socrata…")
            df = fetch_dataset_from_socrata(org, uid, rows)

        if not df.empty:
            # Return info about full dataset and sample
            full_rows = len(df)
            df_sample = df.head(rows)

            # Convert to JSON-serializable format
            data = df_sample.replace({pd.NaT: None}).fillna("").to_dict("records")

            return {
                "sample": data,
                "sample_rows": len(data),
                "total_rows": full_rows,
                "is_sampled": full_rows > rows,
                "columns": list(df.columns),
                "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            }
        else:
            raise HTTPException(status_code=404, detail="Dataset not found")

    except Exception as e:
        logger.error(f"Preview dataset error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets/info")
async def get_dataset_info(
    org: str = Query(...),
    uid: str = Query(...),
):
    """Get dataset info including name for export"""
    try:
        # First check if it's ingested
        ingested_df = sql_df(
            """
            SELECT dataset_uid, dataset_name, org, row_count, column_count
            FROM ingested_datasets
            WHERE dataset_uid = %s AND org = %s
            """,
            params=(uid, org),
        )

        if not ingested_df.empty:
            row = ingested_df.iloc[0]
            return {
                "uid": uid,
                "name": row["dataset_name"] or f"{org} Dataset {uid}",
                "org": org,
                "row_count": row["row_count"],
                "column_count": row["column_count"],
                "is_ingested": True,
            }

        # Not ingested, return basic info
        return {
            "uid": uid,
            "name": f"{org} Dataset {uid}",
            "org": org,
            "is_ingested": False,
        }

    except Exception as e:
        logger.error(f"Get dataset info error: {e}")
        return {
            "uid": uid,
            "name": f"{org} Dataset {uid}",
            "org": org,
            "is_ingested": False,
        }


@app.get("/datasets/quick_preview")
async def quick_preview(
    org: str = Query(...),
    uid: str = Query(...),
    rows: int = Query(200, le=50000),
):
    """Quick preview - alias for preview"""
    return await preview_dataset(org, uid, rows)


# ===========================
# Run the app
# ===========================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
