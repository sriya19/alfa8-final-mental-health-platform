"""
Backend API wrapper for Maryland Mental Health Data Platform
Handles all communication with FastAPI backend
Fixed: Support for all search modes, real stats, and sampling transparency
"""

import os
import requests
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
import logging
import json
import asyncio
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 60


# ---------------------------------------------------------------------
# Health & stats
# ---------------------------------------------------------------------
def check_backend_health() -> Dict[str, Any]:
    """
    Check if backend is healthy

    Returns:
        dict: Health status including 'status' key
    """
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "status": "unhealthy",
                "error": f"Status code: {response.status_code}",
            }
    except Exception as e:
        logger.error(f"Backend health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


def get_database_stats() -> Dict[str, Any]:
    """
    Get real database statistics from backend

    Returns:
        dict: Database statistics including total_chunks, unique_datasets, etc.
    """
    try:
        response = requests.get(f"{BACKEND_URL}/database_stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get database stats: {response.status_code}")
            return {
                "total_chunks": 0,
                "unique_datasets": 0,
                "data_sources": 0,
                "total_records": 0,
                "unique_locations": 0,
                "corpus_breakdown": {"maryland": 0, "ingested": 0},
            }
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {
            "total_chunks": 0,
            "unique_datasets": 0,
            "data_sources": 0,
            "total_records": 0,
            "unique_locations": 0,
            "corpus_breakdown": {"maryland": 0, "ingested": 0},
        }


def get_rag_status(org: Optional[str] = None) -> Dict[str, Any]:
    """
    Get RAG indexing status

    Args:
        org: Optional organization filter

    Returns:
        dict: RAG status including indexed datasets and chunks
    """
    try:
        params = {"org": org} if org else {}
        response = requests.get(f"{BACKEND_URL}/rag_status", params=params, timeout=10)

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "indexed_datasets": 0,
                "total_chunks": 0,
                "ingested_chunks": 0,
                "ingested_dataset_uids": [],
            }
    except Exception as e:
        logger.error(f"Failed to get RAG status: {e}")
        return {
            "indexed_datasets": 0,
            "total_chunks": 0,
            "ingested_chunks": 0,
            "ingested_dataset_uids": [],
        }


# ---------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------
def search_datasets_semantic(
    org: Optional[str],
    query: str,
    limit: int = 10,
    persona: Optional[str] = None,
    filter_ingested_only: bool = False,
) -> Tuple[List[Dict], Optional[str]]:
    """
    Semantic search across all chunks using embeddings

    Args:
        org: Optional organization filter (not required)
        query: Search query/story
        limit: Max results
        persona: Optional user persona
        filter_ingested_only: If True, only search ingested datasets (for RAG)

    Returns:
        Tuple of (results list, error message)
    """
    try:
        payload: Dict[str, Any] = {
            "story": query,
            "k": limit,
            "filter_ingested_only": filter_ingested_only,
        }

        # Only add org if specified (make it optional)
        if org and org != "All":
            payload["org"] = org

        if persona:
            payload["persona"] = persona

        response = requests.post(
            f"{BACKEND_URL}/semantic/search",
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("results", []), None
        else:
            error = f"Semantic search failed: {response.status_code}"
            logger.error(error)
            return [], error

    except Exception as e:
        error = f"Semantic search error: {str(e)}"
        logger.error(error)
        return [], error


def search_datasets_keyword(
    org: str, query: str, limit: int = 10
) -> Tuple[List[Dict], Optional[str]]:
    """
    Keyword search using external Socrata APIs

    Args:
        org: Organization (CDC, SAMHSA, Maryland, or All)
        query: Search query
        limit: Max results

    Returns:
        Tuple of (results list, error message)
    """
    try:
        params = {
            "q": query,
            "org": org,  # Now supports "All"
            "limit": limit,
        }

        response = requests.get(
            f"{BACKEND_URL}/catalog/search",
            params=params,
            timeout=TIMEOUT_SECONDS,
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("results", []), None
        else:
            error = f"Keyword search failed: {response.status_code}"
            logger.error(error)
            return [], error

    except Exception as e:
        error = f"Keyword search error: {str(e)}"
        logger.error(error)
        return [], error


# ---------------------------------------------------------------------
# Ingest / index
# ---------------------------------------------------------------------
def ingest_dataset(org: str, dataset_uid: str, auto_index: bool = False) -> Tuple[bool, str]:
    """
    Ingest a dataset from external source

    Args:
        org: Organization name
        dataset_uid: Dataset identifier
        auto_index: Whether to auto-index for RAG

    Returns:
        Tuple of (success boolean, message)
    """
    try:
        payload = {
            "org": org,
            "dataset_uid": dataset_uid,
            "auto_index": auto_index,
        }

        response = requests.post(
            f"{BACKEND_URL}/ingest",
            json=payload,
            timeout=TIMEOUT_SECONDS * 2,  # Longer timeout for ingestion
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                message = f"✅ Ingested {data.get('rows', 0)} rows"
                if data.get("indexed"):
                    message += f", created {data.get('chunks_created', 0)} chunks"
                return True, message
            else:
                return False, data.get("reason", "Ingestion failed")
        else:
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except Exception:
                error_detail = response.text
            return False, f"Failed: {error_detail}"

    except Exception as e:
        error = f"Ingestion error: {str(e)}"
        logger.error(error)
        return False, error


def index_dataset(org: str, dataset_uid: str) -> Tuple[bool, str]:
    """
    Index a dataset for RAG

    Args:
        org: Organization name
        dataset_uid: Dataset identifier

    Returns:
        Tuple of (success boolean, message)
    """
    try:
        payload = {
            "org": org,
            "dataset_uid": dataset_uid,
        }

        response = requests.post(
            f"{BACKEND_URL}/index_dataset",
            json=payload,
            timeout=TIMEOUT_SECONDS * 2,  # Longer timeout for indexing
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                chunks = data.get("chunks_created", 0)
                return True, f"✅ Created {chunks} searchable chunks"
            else:
                return False, data.get("error", "Indexing failed")
        else:
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except Exception:
                error_detail = response.text
            return False, f"Failed: {error_detail}"

    except Exception as e:
        error = f"Indexing error: {str(e)}"
        logger.error(error)
        return False, error


# ---------------------------------------------------------------------
# Preview / corpus
# ---------------------------------------------------------------------
def preview_dataset(org: str, dataset_uid: str, rows: int = 1000) -> Optional[pd.DataFrame]:
    """
    Preview dataset with transparent sampling info

    Args:
        org: Organization name
        dataset_uid: Dataset identifier
        rows: Max rows to return

    Returns:
        DataFrame or None if error
    """
    try:
        params = {
            "org": org,
            "uid": dataset_uid,
            "rows": rows,
        }

        response = requests.get(
            f"{BACKEND_URL}/datasets/preview",
            params=params,
            timeout=TIMEOUT_SECONDS,
        )

        if response.status_code == 200:
            data = response.json()

            # Log sampling info
            if data.get("is_sampled", False):
                logger.info(
                    f"Dataset sampled: {data.get('sample_rows')} "
                    f"of {data.get('total_rows')} rows"
                )

            df = pd.DataFrame(data.get("sample", []))
            if not df.empty:
                df.attrs["total_rows"] = data.get("total_rows", len(df))
                df.attrs["is_sampled"] = data.get("is_sampled", False)
                df.attrs["sample_rows"] = data.get("sample_rows", len(df))

            return df
        else:
            logger.error(f"Preview failed: {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"Preview error: {str(e)}")
        return None


def get_corpus_text(
    uid: str, max_chunks: int = 25, max_chars: int = 8000
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Call the /corpus/text endpoint to fetch raw corpus text for a semantic dataset.

    Returns:
        (data_dict, error_string)
    """
    try:
        params = {
            "uid": uid,
            "max_chunks": max_chunks,
            "max_chars": max_chars,
        }
        response = requests.get(
            f"{BACKEND_URL}/corpus/text",
            params=params,
            timeout=TIMEOUT_SECONDS,
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            try:
                detail = response.json().get("detail", "Unknown error")
            except Exception:
                detail = response.text
            err = f"Corpus text failed: {response.status_code} - {detail}"
            logger.error(err)
            return None, err
    except Exception as e:
        err = f"Corpus text error: {str(e)}"
        logger.error(err)
        return None, err


# ---------------------------------------------------------------------
# RAG answering
# ---------------------------------------------------------------------
def answer_question(
    question: str,
    org: Optional[str] = None,
    k: int = 5,
    persona: Optional[str] = None,
    ingested_only: bool = True,
    dataset_uids: Optional[List[str]] = None,
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Answer question using RAG

    Args:
        question: User's question
        org: Optional organization filter
        k: Number of context chunks
        persona: Optional user persona
        ingested_only: If True, only use ingested datasets (default)
        dataset_uids: Optional list of dataset_uids to restrict search to

    Returns:
        Tuple of (answer data dict, error message)
    """
    try:
        payload = {
            "question": question,
            "k": k,
            "use_actual_data": True,
            "ingested_only": ingested_only,
        }

        if org and org != "All":
            payload["org"] = org

        if persona:
            payload["persona"] = persona

        if dataset_uids:
            payload["dataset_uids"] = dataset_uids  # ⬅ send list to backend

        response = requests.post(
            f"{BACKEND_URL}/answer",
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )

        if response.status_code == 200:
            return response.json(), None
        else:
            error = f"Failed to answer: {response.status_code}"
            logger.error(error)
            return None, error

    except Exception as e:
        error = f"Answer error: {str(e)}"
        logger.error(error)
        return None, error


# ---------------------------------------------------------------------
# Generic / misc helpers
# ---------------------------------------------------------------------
def call_backend(endpoint: str, method: str = "GET", **kwargs) -> Optional[Dict]:
    """
    Generic backend call wrapper

    Args:
        endpoint: API endpoint path
        method: HTTP method
        **kwargs: Additional arguments for requests

    Returns:
        Response data or None if error
    """
    try:
        url = f"{BACKEND_URL}/{endpoint.lstrip('/')}"
        if method.upper() == "GET":
            response = requests.get(url, **kwargs)
        elif method.upper() == "POST":
            response = requests.post(url, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Backend call failed: {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"Backend call error: {str(e)}")
        return None


def get_dataset_info(org: str, dataset_uid: str) -> Optional[Dict]:
    """
    Get detailed dataset information including size and sampling recommendations

    Args:
        org: Organization name
        dataset_uid: Dataset identifier

    Returns:
        Dataset info dict or None
    """
    try:
        # First get a small preview to check size
        preview_df = preview_dataset(org, dataset_uid, rows=1)

        if preview_df is not None and hasattr(preview_df, "attrs"):
            total_rows = preview_df.attrs.get("total_rows", 0)

            return {
                "uid": dataset_uid,
                "org": org,
                "total_rows": total_rows,
                "columns": list(preview_df.columns),
                "recommended_sample_size": min(1000, total_rows),
                "requires_sampling": total_rows > 5000,
            }

        return None

    except Exception as e:
        logger.error(f"Failed to get dataset info: {e}")
        return None


async def search_all_sources_async(query: str, limit: int = 10) -> List[Dict]:
    """
    Asynchronously search all data sources and combine results

    Args:
        query: Search query
        limit: Max results per source

    Returns:
        Combined and ranked results
    """

    async def search_org(client: httpx.AsyncClient, org: str) -> List[Dict]:
        try:
            response = await client.get(
                f"{BACKEND_URL}/catalog/search",
                params={"q": query, "org": org, "limit": limit},
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                # Add org to each result
                for r in results:
                    r["org"] = org
                return results
            return []
        except Exception as e:
            logger.error(f"Failed to search {org}: {e}")
            return []

    try:
        async with httpx.AsyncClient() as client:
            tasks = [
                search_org(client, "CDC"),
                search_org(client, "SAMHSA"),
                search_org(client, "Maryland"),
            ]

            results_per_org = await asyncio.gather(*tasks)

            # Flatten and combine results
            all_results: List[Dict] = []
            for org_results in results_per_org:
                all_results.extend(org_results)

            # Sort by any relevance score if available
            all_results.sort(
                key=lambda x: x.get("relevance_score", 0), reverse=True
            )

            return all_results[:limit]

    except Exception as e:
        logger.error(f"Async search error: {e}")
        return []


# ---------------------------------------------------------------------
# Exported names
# ---------------------------------------------------------------------
__all__ = [
    "check_backend_health",
    "get_database_stats",
    "get_rag_status",
    "search_datasets_semantic",
    "search_datasets_keyword",
    "ingest_dataset",
    "index_dataset",
    "preview_dataset",
    "answer_question",
    "call_backend",
    "get_dataset_info",
    "search_all_sources_async",
    "get_corpus_text",
]
