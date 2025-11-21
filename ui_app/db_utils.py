"""
Utility helpers for the Streamlit UI.

Provides:
- get_conn()                              -> raw psycopg2 connection
- sql_df(sql, params)                     -> returns a pandas DataFrame
- get_database_stats()                    -> sidebar stats based on ingested_datasets table
- test_connection()                       -> bool
- get_last_error()                        -> last error message (str)

Maryland Corpus Functions:
- get_maryland_corpus_stats()             -> overall Maryland corpus statistics
- get_maryland_corpus_by_source()         -> breakdown by source (CDC, PubMed, etc.)
- get_maryland_corpus_by_location()       -> breakdown by location
- get_maryland_corpus_indexed_stats()     -> indexing status for semantic search
"""

import os
import psycopg2
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

_last_error: str | None = None


def _set_last_error(msg: str) -> None:
    global _last_error
    _last_error = msg


def get_last_error() -> str:
    return _last_error or ""


def get_conn():
    """
    Create a new PostgreSQL connection using env vars.

    Expected env vars (with sensible defaults that match your Docker):
      - POSTGRES_HOST  (default: "localhost")
      - POSTGRES_PORT  (default: "5432")
      - POSTGRES_DB    (default: "mh_catalog")
      - POSTGRES_USER  (default: "app_user")
      - POSTGRES_PASSWORD (default: "app_user")
    """
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    dbname = os.getenv("POSTGRES_DB", "mh_catalog")
    user = os.getenv("POSTGRES_USER", "app_user")
    password = os.getenv("POSTGRES_PASSWORD", "app_user")

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )
        return conn
    except Exception as e:
        _set_last_error(f"Postgres connection failed: {e}")
        raise


def sql_df(sql: str, params: Dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Run a SQL query and return the result as a pandas DataFrame.
    """
    conn = None
    try:
        conn = get_conn()
        df = pd.read_sql_query(sql, conn, params=params)
        return df
    except Exception as e:
        _set_last_error(f"sql_df error: {e}")
        return pd.DataFrame()
    finally:
        if conn is not None:
            conn.close()


def get_database_stats() -> Dict[str, Any]:
    """
    Compute sidebar statistics - FIXED to show actual row counts from ingested_datasets
    """
    stats: Dict[str, Any] = {
        "total_records": 0,
        "unique_datasets": 0,
        "unique_locations": 0,
        "data_sources": 0,
        "by_source": {},
        "date_range": {"start": None, "end": None},
        "total_chunks": 0
    }

    try:
        # 1) Get TOTAL RECORDS - this should be the sum of all rows from all ingested datasets
        records_df = sql_df(
            """
            SELECT 
                COALESCE(SUM(row_count), 0) as total_records
            FROM ingested_datasets
            """,
            {},
        )

        if not records_df.empty and pd.notna(records_df.loc[0, "total_records"]):
            stats["total_records"] = int(records_df.loc[0, "total_records"])
        
        # 2) Get unique datasets and data sources from ingested_datasets
        datasets_df = sql_df(
            """
            SELECT 
                COUNT(DISTINCT dataset_uid) as unique_datasets,
                COUNT(DISTINCT org) as data_sources
            FROM ingested_datasets
            """,
            {},
        )
        
        if not datasets_df.empty:
            stats["unique_datasets"] = int(datasets_df.loc[0, "unique_datasets"])
            stats["data_sources"] = int(datasets_df.loc[0, "data_sources"])

        # 3) Get breakdown by source with actual row counts
        source_df = sql_df(
            """
            SELECT
                org as source,
                SUM(row_count) as cnt
            FROM ingested_datasets
            WHERE row_count IS NOT NULL
            GROUP BY org
            ORDER BY cnt DESC
            """,
            {},
        )

        if not source_df.empty:
            stats["by_source"] = {
                row["source"]: int(row["cnt"])
                for _, row in source_df.iterrows()
                if pd.notna(row["cnt"])
            }

        # 4) Get unique locations (if datasets table exists)
        try:
            location_df = sql_df(
                """
                SELECT COUNT(DISTINCT location) as unique_locations
                FROM datasets
                WHERE location IS NOT NULL AND location != ''
                """,
                {},
            )
            if not location_df.empty and pd.notna(location_df.loc[0, "unique_locations"]):
                stats["unique_locations"] = int(location_df.loc[0, "unique_locations"])
        except:
            # If datasets table doesn't exist, try from ingested_datasets
            stats["unique_locations"] = 0

        # 5) Get total chunks
        chunks_df = sql_df(
            """
            SELECT COUNT(*) as total_chunks
            FROM chunks
            """,
            {},
        )
        
        if not chunks_df.empty and pd.notna(chunks_df.loc[0, "total_chunks"]):
            stats["total_chunks"] = int(chunks_df.loc[0, "total_chunks"])

        # 6) Date range
        date_df = sql_df(
            """
            SELECT
                MIN(ingested_at) AS min_date,
                MAX(ingested_at) AS max_date
            FROM ingested_datasets
            WHERE ingested_at IS NOT NULL
            """,
            {},
        )

        if not date_df.empty:
            if pd.notna(date_df.loc[0, "min_date"]):
                stats["date_range"]["start"] = date_df.loc[0, "min_date"]
            if pd.notna(date_df.loc[0, "max_date"]):
                stats["date_range"]["end"] = date_df.loc[0, "max_date"]

    except Exception as e:
        _set_last_error(f"get_database_stats error: {e}")
        logger.error(f"Database stats error: {e}")

    return stats


def test_connection() -> bool:
    """
    Simple health check for DB connection.
    """
    conn = None
    try:
        conn = get_conn()
        return True
    except Exception:
        return False
    finally:
        if conn is not None:
            conn.close()


# ============================================
# MARYLAND CORPUS QUERY FUNCTIONS
# (For 16k pre-collected Maryland/Baltimore records)
# ============================================

def get_maryland_corpus_stats() -> Dict[str, Any]:
    """
    Get comprehensive statistics about the Maryland corpus.
    
    Returns:
        Dict with: total_records, unique_sources, unique_locations,
                   earliest_date, latest_date
    
    Example:
        stats = get_maryland_corpus_stats()
        st.metric("Maryland Corpus", f"{stats['total_records']:,} records")
    """
    stats = {
        "total_records": 0,
        "unique_sources": 0,
        "unique_locations": 0,
        "earliest_date": None,
        "latest_date": None
    }
    
    try:
        result_df = sql_df("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT source) as unique_sources,
                COUNT(DISTINCT location) as unique_locations,
                MIN(created_at) as earliest_date,
                MAX(created_at) as latest_date
            FROM maryland_mental_health_data
        """, {})
        
        if not result_df.empty:
            row = result_df.iloc[0]
            stats["total_records"] = int(row["total_records"]) if pd.notna(row["total_records"]) else 0
            stats["unique_sources"] = int(row["unique_sources"]) if pd.notna(row["unique_sources"]) else 0
            stats["unique_locations"] = int(row["unique_locations"]) if pd.notna(row["unique_locations"]) else 0
            stats["earliest_date"] = row["earliest_date"]
            stats["latest_date"] = row["latest_date"]
            
    except Exception as e:
        _set_last_error(f"get_maryland_corpus_stats error: {e}")
    
    return stats


def get_maryland_corpus_by_source() -> pd.DataFrame:
    """
    Get record count grouped by source (CDC, PubMed, Maryland Open Data, etc.).
    
    Returns:
        DataFrame with columns: source, count
        
    Example:
        df = get_maryland_corpus_by_source()
        st.bar_chart(df.set_index('source'))
    """
    try:
        return sql_df("""
            SELECT 
                source,
                COUNT(*) as count
            FROM maryland_mental_health_data
            GROUP BY source
            ORDER BY count DESC
        """, {})
    except Exception as e:
        _set_last_error(f"get_maryland_corpus_by_source error: {e}")
        return pd.DataFrame(columns=["source", "count"])


def get_maryland_corpus_by_location() -> pd.DataFrame:
    """
    Get record count grouped by location.
    
    Returns:
        DataFrame with columns: location, count
        
    Example:
        df = get_maryland_corpus_by_location()
        st.dataframe(df)
    """
    try:
        return sql_df("""
            SELECT 
                location,
                COUNT(*) as count
            FROM maryland_mental_health_data
            WHERE location IS NOT NULL AND location != ''
            GROUP BY location
            ORDER BY count DESC
            LIMIT 20
        """, {})
    except Exception as e:
        _set_last_error(f"get_maryland_corpus_by_location error: {e}")
        return pd.DataFrame(columns=["location", "count"])


def get_maryland_corpus_indexed_stats() -> Dict[str, int]:
    """
    Get statistics about Maryland corpus chunks indexed for semantic search.
    
    Returns how many Maryland corpus records have been indexed into
    the chunks table for RAG/semantic search.
    
    Returns:
        Dict with: total_chunks, indexed_datasets
        
    Example:
        stats = get_maryland_corpus_indexed_stats()
        st.info(f"Indexed: {stats['total_chunks']} chunks from Maryland corpus")
    """
    stats = {
        "total_chunks": 0,
        "indexed_datasets": 0
    }
    
    try:
        result_df = sql_df("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(DISTINCT dataset_uid) as indexed_datasets
            FROM chunks
            WHERE chunk_id LIKE 'MD_%' 
               OR dataset_uid LIKE 'maryland_data_%'
        """, {})
        
        if not result_df.empty:
            row = result_df.iloc[0]
            stats["total_chunks"] = int(row["total_chunks"]) if pd.notna(row["total_chunks"]) else 0
            stats["indexed_datasets"] = int(row["indexed_datasets"]) if pd.notna(row["indexed_datasets"]) else 0
            
    except Exception as e:
        _set_last_error(f"get_maryland_corpus_indexed_stats error: {e}")
    
    return stats


def get_ingested_datasets_stats() -> Dict[str, Any]:
    """
    Get statistics about UI-ingested datasets.
    
    Returns:
        Dict with: total_ingested, total_indexed, total_rows
        
    Example:
        stats = get_ingested_datasets_stats()
        st.metric("Ingested Datasets", stats['total_ingested'])
    """
    stats = {
        "total_ingested": 0,
        "total_indexed": 0,
        "total_rows": 0
    }
    
    try:
        result_df = sql_df("""
            SELECT 
                COUNT(*) as total_ingested,
                COUNT(*) FILTER (WHERE indexed = TRUE) as total_indexed,
                COALESCE(SUM(row_count), 0) as total_rows
            FROM ingested_datasets
        """, {})
        
        if not result_df.empty:
            row = result_df.iloc[0]
            stats["total_ingested"] = int(row["total_ingested"]) if pd.notna(row["total_ingested"]) else 0
            stats["total_indexed"] = int(row["total_indexed"]) if pd.notna(row["total_indexed"]) else 0
            stats["total_rows"] = int(row["total_rows"]) if pd.notna(row["total_rows"]) else 0
            
    except Exception as e:
        _set_last_error(f"get_ingested_datasets_stats error: {e}")
    
    return stats


def get_chunks_stats() -> Dict[str, Any]:
    """
    Get comprehensive statistics about indexed chunks.
    
    Breaks down chunks by:
    - UI-ingested datasets vs Maryland corpus
    - Organization
    
    Returns:
        Dict with: total_chunks, ingested_chunks, maryland_chunks, by_org
        
    Example:
        stats = get_chunks_stats()
        st.metric("Total Chunks", f"{stats['total_chunks']:,}")
        st.caption(f"Ingested: {stats['ingested_chunks']}, Maryland: {stats['maryland_chunks']}")
    """
    stats = {
        "total_chunks": 0,
        "ingested_chunks": 0,
        "maryland_chunks": 0,
        "by_org": {}
    }
    
    try:
        # Total chunks
        total_df = sql_df("SELECT COUNT(*) as count FROM chunks", {})
        if not total_df.empty:
            stats["total_chunks"] = int(total_df.iloc[0]["count"])
        
        # Breakdown by corpus type
        corpus_df = sql_df("""
            SELECT 
                CASE 
                    WHEN chunk_id LIKE 'MD_%' OR dataset_uid LIKE 'maryland_data_%' 
                    THEN 'maryland'
                    ELSE 'ingested'
                END as corpus_type,
                COUNT(*) as count
            FROM chunks
            GROUP BY corpus_type
        """, {})
        
        for _, row in corpus_df.iterrows():
            if row["corpus_type"] == "maryland":
                stats["maryland_chunks"] = int(row["count"])
            else:
                stats["ingested_chunks"] = int(row["count"])
        
        # By organization
        org_df = sql_df("""
            SELECT org, COUNT(*) as count
            FROM chunks
            WHERE org IS NOT NULL
            GROUP BY org
            ORDER BY count DESC
        """, {})
        
        stats["by_org"] = {
            row["org"]: int(row["count"])
            for _, row in org_df.iterrows()
            if pd.notna(row["org"])
        }
        
    except Exception as e:
        _set_last_error(f"get_chunks_stats error: {e}")
    
    return stats


def search_maryland_corpus_text(
    query: str, 
    limit: int = 100,
    source_filter: str = None
) -> pd.DataFrame:
    """
    Perform simple text search in maryland_mental_health_data.
    
    Note: This is basic text search. For semantic search,
          use the backend API with chunks/embeddings.
    
    Args:
        query: Search term
        limit: Maximum results to return
        source_filter: Optional filter by source (e.g., 'CDC', 'PubMed')
        
    Returns:
        DataFrame with columns: id, source, title, location, content_preview, url
        
    Example:
        results = search_maryland_corpus_text("depression", limit=50)
        st.dataframe(results)
    """
    try:
        if source_filter:
            return sql_df("""
                SELECT 
                    id, 
                    source, 
                    title, 
                    location, 
                    LEFT(content, 500) as content_preview,
                    url
                FROM maryland_mental_health_data
                WHERE (
                    title ILIKE %(query)s 
                    OR content ILIKE %(query)s
                )
                AND source = %(source_filter)s
                ORDER BY created_at DESC
                LIMIT %(limit)s
            """, {
                "query": f"%{query}%",
                "source_filter": source_filter,
                "limit": limit
            })
        else:
            return sql_df("""
                SELECT 
                    id, 
                    source, 
                    title, 
                    location, 
                    LEFT(content, 500) as content_preview,
                    url
                FROM maryland_mental_health_data
                WHERE (
                    title ILIKE %(query)s 
                    OR content ILIKE %(query)s
                )
                ORDER BY created_at DESC
                LIMIT %(limit)s
            """, {
                "query": f"%{query}%",
                "limit": limit
            })
    except Exception as e:
        _set_last_error(f"search_maryland_corpus_text error: {e}")
        return pd.DataFrame(columns=["id", "source", "title", "location", "content_preview", "url"])