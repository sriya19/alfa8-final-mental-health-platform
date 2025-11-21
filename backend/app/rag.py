"""
RAG (Retrieval-Augmented Generation) Module
Creates searchable chunks with embeddings for semantic search
Works with:
1. UI-ingested datasets from mh-raw bucket
2. Maryland corpus (16k pre-collected records)
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import openai
import psycopg2
from psycopg2.extras import Json
import traceback

logger = logging.getLogger(__name__)

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    logger.info("OpenAI API key configured")
else:
    logger.warning("No OpenAI API key found - embeddings will be random")


def create_embedding(text: str) -> List[float]:
    """
    Create embedding using OpenAI's text-embedding-3-small model with retry logic.
    If no API key is configured or all retries fail, returns a random 1536-dim vector.
    """
    if not OPENAI_API_KEY:
        logger.warning("No OpenAI API key, returning random embedding")
        return np.random.rand(1536).tolist()

    # Truncate text to avoid token limits
    text = text[:8000]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                timeout=60.0,  # Increased timeout to 60 seconds
            )
            embedding = response.data[0].embedding
            logger.info(f"Created embedding for text of length {len(text)}")
            return embedding

        except Exception as e:
            error_type = type(e).__name__
            logger.warning(
                f"Embedding attempt {attempt + 1}/{max_retries} failed ({error_type}): {e}"
            )

            if attempt < max_retries - 1:
                import time

                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to create embedding after {max_retries} attempts")
                # Return random embedding as fallback
                logger.warning("Using random embedding as fallback")
                return np.random.rand(1536).tolist()

    # Fallback (should never reach here)
    return np.random.rand(1536).tolist()


def create_chunks_from_dataframe(df: pd.DataFrame, chunk_size: int = 50) -> List[Dict[str, Any]]:
    """
    Convert DataFrame into searchable chunks with rich context

    Strategy:
    1. Try intelligent grouping by common columns (state, year, category, etc.)
    2. Create statistical summaries for each group
    3. Include sample records
    4. Fall back to sequential chunks if grouping yields too few chunks

    Args:
        df: DataFrame to chunk
        chunk_size: Number of rows per chunk for sequential chunking (default: 50)

    Returns:
        List of chunk dictionaries with 'content' and 'metadata'
    """
    chunks = []
    logger.info(
        f"Creating chunks from DataFrame with {len(df)} rows and {len(df.columns)} columns"
    )

    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    logger.info(f"Found {len(numeric_cols)} numeric columns and {len(text_cols)} text columns")

    # Try intelligent grouping first
    group_cols: List[str] = []
    potential_group_cols = [
        "state",
        "location",
        "locationabbr",
        "year",
        "category",
        "indicator",
        "county",
        "stratification1",
        "topic",
        "class",
        "question",
        "response",
        "data_value_type",
        "geolocation",
    ]

    for col in potential_group_cols:
        matching_cols = [c for c in df.columns if col in c.lower()]
        if matching_cols:
            group_cols.extend(matching_cols[:1])  # Take first match
            if len(group_cols) >= 2:
                break

    if group_cols:
        logger.info(f"Grouping by columns: {group_cols}")
        try:
            # Group by available columns
            grouped = df.groupby(group_cols[: min(2, len(group_cols))])

            for group_key, group_df in grouped:
                if len(group_df) == 0:
                    continue

                # Create comprehensive chunk text
                chunk_lines: List[str] = []

                # Add group information
                if isinstance(group_key, tuple):
                    group_info = ", ".join(
                        [f"{col}: {val}" for col, val in zip(group_cols[:2], group_key)]
                    )
                else:
                    group_info = f"{group_cols[0]}: {group_key}"

                chunk_lines.append(f"Data Group: {group_info}")
                chunk_lines.append(f"Number of records: {len(group_df)}")
                chunk_lines.append("")

                # Add statistics for numeric columns
                for col in numeric_cols[:10]:
                    if col in group_df.columns and group_df[col].notna().any():
                        try:
                            stats = group_df[col].describe()
                            chunk_lines.append(f"{col}:")
                            chunk_lines.append(f"  Mean: {stats['mean']:.2f}")
                            chunk_lines.append(
                                f"  Min: {stats['min']:.2f}, Max: {stats['max']:.2f}"
                            )
                            chunk_lines.append(f"  Count: {stats['count']:.0f}")
                        except Exception:
                            pass

                # Add categorical value counts (cast to str to avoid unhashable dict)
                for col in text_cols[:5]:
                    if col not in group_cols and col in group_df.columns:
                        try:
                            vc = group_df[col].astype(str).value_counts().head(5)
                            if not vc.empty:
                                chunk_lines.append(f"{col} distribution:")
                                for val, cnt in vc.items():
                                    chunk_lines.append(f"  {val}: {cnt}")
                        except Exception:
                            pass

                # Add sample records
                chunk_lines.append("\nSample records:")
                for idx, (_, row) in enumerate(group_df.head(3).iterrows()):
                    row_text = " | ".join(
                        [
                            f"{k}: {v}"
                            for k, v in row.items()
                            if pd.notna(v) and str(v).strip()
                        ][:10]
                    )
                    chunk_lines.append(f"  Record {idx + 1}: {row_text}")

                chunk_text = "\n".join(chunk_lines)

                # Create metadata
                if isinstance(group_key, tuple):
                    group_dict = dict(zip(group_cols[:2], group_key))
                else:
                    group_dict = {group_cols[0]: group_key}

                metadata = {
                    "group": group_dict,
                    "row_count": len(group_df),
                    "columns": group_df.columns.tolist()[:20],
                }

                chunks.append({"content": chunk_text, "metadata": metadata})

                if len(chunks) >= 100:
                    logger.info("Reached chunk limit of 100")
                    break

        except Exception as e:
            logger.error(f"Error creating grouped chunks: {e}")

    # If not enough chunks from grouping, create sequential chunks
    if len(chunks) < 20:
        logger.info(f"Creating additional sequential chunks (current: {len(chunks)})")

        for i in range(0, min(len(df), 2000), chunk_size):
            chunk_df = df.iloc[i : i + chunk_size]

            chunk_lines = [
                f"Dataset records {i + 1} to {min(i + chunk_size, len(df))}",
                f"Total records in chunk: {len(chunk_df)}",
                "",
            ]

            # Add column summaries
            for col in chunk_df.columns[:15]:
                if col in numeric_cols:
                    try:
                        mean_val = chunk_df[col].mean()
                        if pd.notna(mean_val):
                            chunk_lines.append(f"{col}: {mean_val:.2f} (average)")
                    except Exception:
                        pass
                elif col in text_cols:
                    # Cast to str to avoid "unhashable type: 'dict'"
                    try:
                        unique_vals = chunk_df[col].astype(str).nunique()
                        chunk_lines.append(f"{col}: {unique_vals} unique values")
                    except Exception:
                        pass

            # Add full sample records
            chunk_lines.append("\nComplete sample data:")
            for _, row in chunk_df.head(5).iterrows():
                row_text = " | ".join(
                    [
                        f"{k}: {v}"
                        for k, v in row.items()
                        if pd.notna(v) and str(v).strip()
                    ][:12]
                )
                chunk_lines.append(f"  {row_text}")

            chunk_text = "\n".join(chunk_lines)

            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        "row_range": [i, min(i + chunk_size, len(df))],
                        "columns": chunk_df.columns.tolist()[:20],
                    },
                }
            )

            if len(chunks) >= 100:
                break

    logger.info(f"Created {len(chunks)} chunks total")
    return chunks


def index_dataset_for_rag(org: str, dataset_uid: str, s3_key: str = None) -> Dict[str, Any]:
    """
    Index a UI-ingested dataset for RAG by creating searchable chunks with embeddings

    NOTE: This is for UI-ingested datasets only. Maryland corpus uses
          separate indexing script (index_maryland_for_rag.py)
    """
    from app.ingest import get_dataset_from_minio

    logger.info(f"Starting RAG indexing for UI-ingested dataset {org}/{dataset_uid}")

    # Database configuration
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "pg"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "database": os.getenv("POSTGRES_DB", "mh_catalog"),
        "user": os.getenv("POSTGRES_USER", "app_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "app_user"),
    }

    conn = None
    try:
        # Get dataset from MinIO
        df = get_dataset_from_minio(org, dataset_uid)

        if df.empty:
            error_msg = f"Dataset not found or empty for {org}/{dataset_uid}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

        logger.info(f"Retrieved dataset with {len(df)} rows")

        # Create chunks
        chunks = create_chunks_from_dataframe(df)

        if not chunks:
            error_msg = "No chunks created from dataset"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

        # Store chunks in PostgreSQL
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Create chunks table if not exists
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id BIGSERIAL PRIMARY KEY,
                dataset_uid TEXT NOT NULL,
                org TEXT NOT NULL,
                chunk_id TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding vector(1536),
                created_at TIMESTAMP DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS chunks_dataset_idx ON chunks(org, dataset_uid);
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        """
        )

        # Delete old chunks for this dataset
        cursor.execute(
            """
            DELETE FROM chunks
            WHERE org = %s AND dataset_uid = %s
        """,
            (org, dataset_uid),
        )

        deleted_count = cursor.rowcount
        logger.info(f"Deleted {deleted_count} old chunks for {org}/{dataset_uid}")

        # Insert new chunks with embeddings
        inserted_count = 0
        for i, chunk in enumerate(chunks):
            # Chunk ID format for UI-ingested datasets: {org}_{dataset_uid}_{index}
            chunk_id = f"{org}_{dataset_uid}_{i}"

            # Create embedding for chunk content
            logger.info(f"Creating embedding for chunk {i + 1}/{len(chunks)}")
            embedding = create_embedding(chunk["content"][:2000])

            # Format embedding for PostgreSQL vector type
            embedding_str = f"[{','.join(map(str, embedding))}]"

            try:
                cursor.execute(
                    """
                    INSERT INTO chunks
                    (dataset_uid, org, chunk_id, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                """,
                    (
                        dataset_uid,
                        org,
                        chunk_id,
                        chunk["content"],
                        Json(chunk["metadata"]),
                        embedding_str,
                    ),
                )
                inserted_count += 1

            except Exception as e:
                logger.error(f"Failed to insert chunk {chunk_id}: {e}")
                continue

        # Update dataset as indexed
        cursor.execute(
            """
            UPDATE ingested_datasets
            SET indexed = TRUE
            WHERE org = %s AND dataset_uid = %s
        """,
            (org, dataset_uid),
        )

        conn.commit()

        logger.info(
            f"✅ Successfully indexed {inserted_count} chunks for {org}/{dataset_uid}"
        )

        return {
            "success": True,
            "chunks_created": inserted_count,
            "dataset_uid": dataset_uid,
            "org": org,
        }

    except Exception as e:
        logger.error(f"Failed to index dataset: {e}")
        logger.error(f"Full error: {traceback.format_exc()}")
        if conn:
            conn.rollback()
        return {"success": False, "error": str(e)}
    finally:
        if conn:
            cursor.close()
            conn.close()


def search_chunks(
    query: str,
    org: Optional[str] = None,
    limit: int = 10,
    corpus_types: Optional[List[str]] = None,
    dataset_uids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search for relevant chunks using semantic similarity (vector search).

    Supports filters:
    - org: optional organization filter
    - corpus_types: ['ingested'], ['maryland'], ['ingested','maryland'] or None/'all'
    - dataset_uids: optional list of dataset_uids to restrict RAG to specific dataset(s)

    This version does NOT use any %s placeholders in the SQL,
    so there is no way to get a "tuple index out of range" from params.
    """
    logger.info(
        f"Searching chunks for query: '{query[:100]}...' "
        f"(org={org}, limit={limit}, corpus_types={corpus_types}, dataset_uids={dataset_uids})"
    )

    db_config = {
        "host": os.getenv("POSTGRES_HOST", "pg"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "database": os.getenv("POSTGRES_DB", "mh_catalog"),
        "user": os.getenv("POSTGRES_USER", "app_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "app_user"),
    }

    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # -----------------------
        # Build WHERE clause
        # -----------------------
        where_clauses: List[str] = []

        # org filter
        if org:
            safe_org = org.replace("'", "''")
            where_clauses.append(f"org = '{safe_org}'")

        # corpus type filter
        if corpus_types and corpus_types != ["all"]:
            corpus_conditions: List[str] = []
            if "ingested" in corpus_types:
                corpus_conditions.append(
                    "(chunk_id NOT LIKE 'MD_%' AND dataset_uid NOT LIKE 'maryland_data_%')"
                )
            if "maryland" in corpus_types:
                corpus_conditions.append(
                    "(chunk_id LIKE 'MD_%' OR dataset_uid LIKE 'maryland_data_%')"
                )
            if corpus_conditions:
                where_clauses.append(f"({' OR '.join(corpus_conditions)})")

        # dataset filter
        if dataset_uids:
            safe_uids = [u.replace("'", "''") for u in dataset_uids]
            uid_list_sql = ", ".join(f"'{u}'" for u in safe_uids)
            where_clauses.append(f"dataset_uid IN ({uid_list_sql})")

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # -----------------------
        # Count chunks (for logging)
        # -----------------------
        count_sql = f"SELECT COUNT(*) FROM chunks WHERE {where_sql}"
        logger.info(f"Chunk count SQL:\n{count_sql}")
        cursor.execute(count_sql)
        total_chunks = cursor.fetchone()[0]
        logger.info(f"Total chunks matching filter: {total_chunks}")

        if total_chunks == 0:
            logger.warning("No chunks found with current filters - semantic search unavailable")
            return []

        # -----------------------
        # Create embedding for query
        # -----------------------
        query_embedding = create_embedding(query)
        embedding_str = f"[{','.join(map(str, query_embedding))}]"

        # -----------------------
        # Vector similarity search
        # -----------------------
        limit_int = int(limit)

        query_sql = f"""
            SELECT
                chunk_id,
                dataset_uid,
                org,
                content,
                metadata,
                1 - (embedding <=> '{embedding_str}'::vector) AS similarity
            FROM chunks
            WHERE {where_sql}
            ORDER BY embedding <=> '{embedding_str}'::vector
            LIMIT {limit_int}
        """

        logger.info(f"Vector search SQL:\n{query_sql}")
        cursor.execute(query_sql)  # no params → no tuple index mismatch

        results: List[Dict[str, Any]] = []
        for row in cursor.fetchall():
            chunk_id = row[0]
            dataset_uid = row[1]

            if chunk_id.startswith("MD_") or dataset_uid.startswith("maryland_data_"):
                corpus_type = "maryland"
            else:
                corpus_type = "ingested"

            results.append(
                {
                    "chunk_id": chunk_id,
                    "dataset_uid": dataset_uid,
                    "org": row[2],
                    "content": row[3],
                    "metadata": row[4],
                    "similarity": float(row[5]),
                    "corpus_type": corpus_type,
                }
            )

        logger.info(f"Found {len(results)} matching chunks")

        ingested_count = sum(1 for r in results if r["corpus_type"] == "ingested")
        maryland_count = sum(1 for r in results if r["corpus_type"] == "maryland")
        logger.info(f"Results: {ingested_count} ingested, {maryland_count} Maryland corpus")

        if results:
            logger.info(
                f"Top result similarity: {results[0]['similarity']:.3f} "
                f"from {results[0]['corpus_type']} ({results[0]['dataset_uid']})"
            )

        return results

    except Exception as e:
        logger.error(f"Search failed: {e}")
        logger.error(f"Full error: {traceback.format_exc()}")
        return []
    finally:
        if conn:
            cursor.close()
            conn.close()


def get_chunk_by_id(chunk_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific chunk by its ID
    """
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "pg"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "database": os.getenv("POSTGRES_DB", "mh_catalog"),
        "user": os.getenv("POSTGRES_USER", "app_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "app_user"),
    }

    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT chunk_id, dataset_uid, org, content, metadata, created_at
            FROM chunks
            WHERE chunk_id = %s
        """,
            (chunk_id,),
        )

        row = cursor.fetchone()

        if row:
            return {
                "chunk_id": row[0],
                "dataset_uid": row[1],
                "org": row[2],
                "content": row[3],
                "metadata": row[4],
                "created_at": row[5],
            }

        return None

    except Exception as e:
        logger.error(f"Failed to get chunk: {e}")
        return None
    finally:
        if conn:
            cursor.close()
            conn.close()


def get_dataset_chunks(org: str, dataset_uid: str) -> List[Dict[str, Any]]:
    """
    Get all chunks for a specific dataset
    """
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "pg"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "database": os.getenv("POSTGRES_DB", "mh_catalog"),
        "user": os.getenv("POSTGRES_USER", "app_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "app_user"),
    }

    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT chunk_id, content, metadata, created_at
            FROM chunks
            WHERE org = %s AND dataset_uid = %s
            ORDER BY chunk_id
        """,
            (org, dataset_uid),
        )

        results: List[Dict[str, Any]] = []
        for row in cursor.fetchall():
            results.append(
                {
                    "chunk_id": row[0],
                    "content": row[1],
                    "metadata": row[2],
                    "created_at": row[3],
                }
            )

        return results

    except Exception as e:
        logger.error(f"Failed to get dataset chunks: {e}")
        return []
    finally:
        if conn:
            cursor.close()
            conn.close()


def delete_dataset_chunks(org: str, dataset_uid: str) -> Dict[str, Any]:
    """
    Delete all chunks for a specific dataset
    """
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "pg"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "database": os.getenv("POSTGRES_DB", "mh_catalog"),
        "user": os.getenv("POSTGRES_USER", "app_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "app_user"),
    }

    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM chunks
            WHERE org = %s AND dataset_uid = %s
        """,
            (org, dataset_uid),
        )

        deleted_count = cursor.rowcount

        # Update dataset as not indexed
        cursor.execute(
            """
            UPDATE ingested_datasets
            SET indexed = FALSE
            WHERE org = %s AND dataset_uid = %s
        """,
            (org, dataset_uid),
        )

        conn.commit()

        logger.info(f"✅ Deleted {deleted_count} chunks for {org}/{dataset_uid}")

        return {"success": True, "deleted_count": deleted_count}

    except Exception as e:
        logger.error(f"Failed to delete chunks: {e}")
        if conn:
            conn.rollback()
        return {"success": False, "error": str(e)}
    finally:
        if conn:
            cursor.close()
            conn.close()


def get_indexing_stats() -> Dict[str, Any]:
    """
    Get statistics about indexed datasets and chunks

    Includes stats for BOTH:
    - UI-ingested datasets
    - Maryland corpus
    """
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "pg"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "database": os.getenv("POSTGRES_DB", "mh_catalog"),
        "user": os.getenv("POSTGRES_USER", "app_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "app_user"),
    }

    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Total chunks
        cursor.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]

        # Chunks by corpus type
        cursor.execute(
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
        """
        )
        corpus_breakdown = {row[0]: row[1] for row in cursor.fetchall()}

        # Unique datasets
        cursor.execute("SELECT COUNT(DISTINCT dataset_uid) FROM chunks")
        indexed_datasets = cursor.fetchone()[0]

        # Chunks by org
        cursor.execute(
            """
            SELECT org, COUNT(*) as chunk_count
            FROM chunks
            GROUP BY org
            ORDER BY chunk_count DESC
        """
        )
        by_org = {row[0]: row[1] for row in cursor.fetchall()}

        # Recent indexing
        cursor.execute(
            """
            SELECT dataset_uid, org, COUNT(*) as chunk_count, MAX(created_at) as last_indexed
            FROM chunks
            GROUP BY dataset_uid, org
            ORDER BY MAX(created_at) DESC
            LIMIT 5
        """
        )
        recent = []
        for row in cursor.fetchall():
            recent.append(
                {
                    "dataset_uid": row[0],
                    "org": row[1],
                    "chunk_count": row[2],
                    "last_indexed": row[3],
                }
            )

        return {
            "total_chunks": total_chunks,
            "ingested_chunks": corpus_breakdown.get("ingested", 0),
            "maryland_chunks": corpus_breakdown.get("maryland", 0),
            "indexed_datasets": indexed_datasets,
            "by_org": by_org,
            "recent": recent,
        }

    except Exception as e:
        logger.error(f"Failed to get indexing stats: {e}")
        return {
            "total_chunks": 0,
            "ingested_chunks": 0,
            "maryland_chunks": 0,
            "indexed_datasets": 0,
            "by_org": {},
            "recent": [],
        }
    finally:
        if conn:
            cursor.close()
            conn.close()


# Utility function for testing
def test_rag_indexing():
    """Test function to verify RAG indexing and search"""
    print("Testing RAG indexing and search...")

    # First, you need an ingested dataset
    # For this test, we'll assume you have one
    test_org = "CDC"
    test_uid = "hyst-znpv"  # Small test dataset

    print(f"\n1. Testing indexing: {test_org}/{test_uid}")
    result = index_dataset_for_rag(test_org, test_uid)

    if result["success"]:
        print("✅ Indexing successful!")
        print(f"   Chunks created: {result['chunks_created']}")
    else:
        print(f"❌ Indexing failed: {result.get('error', 'Unknown error')}")
        return

    print("\n2. Testing semantic search (all corpus)")
    chunks = search_chunks("mental health data", limit=3)

    if chunks:
        print(f"✅ Found {len(chunks)} relevant chunks")
        for chunk in chunks:
            print(f"   - {chunk['corpus_type']}: similarity={chunk['similarity']:.3f}")
    else:
        print("❌ No chunks found")

    print("\n3. Testing Maryland corpus only")
    md_chunks = search_chunks("Baltimore", corpus_types=["maryland"], limit=3)
    print(f"Found {len(md_chunks)} Maryland corpus chunks")

    print("\n4. Getting indexing stats")
    stats = get_indexing_stats()
    print(f"✅ Total chunks: {stats['total_chunks']}")
    print(f"✅ Ingested: {stats['ingested_chunks']}, Maryland: {stats['maryland_chunks']}")

    print("\n✅ RAG tests complete!")


if __name__ == "__main__":
    # Run tests if executed directly
    test_rag_indexing()
