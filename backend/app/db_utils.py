import os
from typing import Any, Dict, Optional, List

import pandas as pd
import psycopg2


def get_db_config() -> Dict[str, Any]:
    """Return DB config using the same env vars as the backend/ingest."""
    return {
        "host": os.getenv("POSTGRES_HOST", "pg"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "database": os.getenv("POSTGRES_DB", "mh_catalog"),
        "user": os.getenv("POSTGRES_USER", "app_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "app_user"),
    }


def get_connection():
    """Open a new psycopg2 connection."""
    return psycopg2.connect(**get_db_config())


def sql_df(sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """
    Run a SQL query and return a pandas DataFrame.
    Used by rag_status and other backend endpoints.
    """
    conn = get_connection()
    try:
        df = pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()
    return df


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
        print(f"Total records: {stats['total_records']}")
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Get comprehensive stats in one query
        cursor.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT source) as unique_sources,
                COUNT(DISTINCT location) as unique_locations,
                MIN(created_at) as earliest_date,
                MAX(created_at) as latest_date
            FROM maryland_mental_health_data
        """)
        
        row = cursor.fetchone()
        
        if row:
            return {
                "total_records": row[0] or 0,
                "unique_sources": row[1] or 0,
                "unique_locations": row[2] or 0,
                "earliest_date": row[3],
                "latest_date": row[4]
            }
        else:
            return {
                "total_records": 0,
                "unique_sources": 0,
                "unique_locations": 0,
                "earliest_date": None,
                "latest_date": None
            }
    finally:
        cursor.close()
        conn.close()


def get_maryland_corpus_by_source() -> pd.DataFrame:
    """
    Get record count grouped by source (CDC, PubMed, Maryland Open Data, etc.).
    
    Returns:
        DataFrame with columns: source, count
        
    Example:
        df = get_maryland_corpus_by_source()
        # source              count
        # CDC                 5432
        # PubMed              3210
        # Maryland Open Data  7890
    """
    return sql_df("""
        SELECT 
            source,
            COUNT(*) as count
        FROM maryland_mental_health_data
        GROUP BY source
        ORDER BY count DESC
    """)


def get_maryland_corpus_by_location() -> pd.DataFrame:
    """
    Get record count grouped by location.
    
    Returns:
        DataFrame with columns: location, count
        
    Example:
        df = get_maryland_corpus_by_location()
        # location      count
        # Baltimore     4521
        # Montgomery    1234
        # Maryland      8765
    """
    return sql_df("""
        SELECT 
            location,
            COUNT(*) as count
        FROM maryland_mental_health_data
        WHERE location IS NOT NULL AND location != ''
        GROUP BY location
        ORDER BY count DESC
        LIMIT 20
    """)


def get_maryland_record_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific record from maryland_mental_health_data by ID.
    
    Args:
        record_id: The record ID to retrieve
        
    Returns:
        Dict with record data or None if not found
        
    Example:
        record = get_maryland_record_by_id(123)
        if record:
            print(record['title'])
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                id, source, dataset_id, title, content, 
                location, metadata, url, created_at
            FROM maryland_mental_health_data
            WHERE id = %s
        """, (record_id,))
        
        row = cursor.fetchone()
        
        if row:
            return {
                "id": row[0],
                "source": row[1],
                "dataset_id": row[2],
                "title": row[3],
                "content": row[4],
                "location": row[5],
                "metadata": row[6],
                "url": row[7],
                "created_at": row[8]
            }
        return None
        
    finally:
        cursor.close()
        conn.close()


def search_maryland_corpus_text(
    query: str, 
    limit: int = 100,
    source_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Perform full-text search in maryland_mental_health_data.
    
    Note: This is a simple text search. For semantic search,
          use the chunks table with embeddings via RAG.
    
    Args:
        query: Search term
        limit: Maximum results to return
        source_filter: Optional filter by source (e.g., 'CDC', 'PubMed')
        
    Returns:
        List of matching records
        
    Example:
        results = search_maryland_corpus_text("depression", limit=50)
        results = search_maryland_corpus_text("suicide", source_filter="CDC")
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Build query with optional source filter
        sql = """
            SELECT 
                id, source, title, location, 
                LEFT(content, 500) as content_preview,
                url
            FROM maryland_mental_health_data
            WHERE (
                title ILIKE %s 
                OR content ILIKE %s
            )
        """
        
        params = [f"%{query}%", f"%{query}%"]
        
        if source_filter:
            sql += " AND source = %s"
            params.append(source_filter)
        
        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(sql, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "source": row[1],
                "title": row[2],
                "location": row[3],
                "content_preview": row[4],
                "url": row[5]
            })
        
        return results
        
    finally:
        cursor.close()
        conn.close()


def get_maryland_corpus_indexed_stats() -> Dict[str, Any]:
    """
    Get statistics about Maryland corpus chunks indexed for semantic search.
    
    Returns how many Maryland corpus records have been indexed into
    the chunks table for RAG/semantic search.
    
    Returns:
        Dict with: total_chunks, indexed_records, avg_chunks_per_record
        
    Example:
        stats = get_maryland_corpus_indexed_stats()
        print(f"Indexed chunks: {stats['total_chunks']}")
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Count chunks that came from Maryland corpus
        # (Assumes chunk_id pattern like 'MD_*' or dataset_uid like 'maryland_data_*')
        cursor.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(DISTINCT dataset_uid) as unique_datasets
            FROM chunks
            WHERE chunk_id LIKE 'MD_%' 
               OR dataset_uid LIKE 'maryland_data_%'
        """)
        
        row = cursor.fetchone()
        
        if row:
            return {
                "total_chunks": row[0] or 0,
                "indexed_datasets": row[1] or 0
            }
        else:
            return {
                "total_chunks": 0,
                "indexed_datasets": 0
            }
            
    finally:
        cursor.close()
        conn.close()