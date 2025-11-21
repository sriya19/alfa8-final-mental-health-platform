#!/usr/bin/env python3
"""
Index maryland_mental_health_data table for RAG semantic search
Converts the Maryland records into searchable chunks with embeddings.
"""

import os
import sys
import time
import json
import logging
from typing import List, Dict, Optional

import psycopg2
from psycopg2.extras import Json
from openai import OpenAI

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# OpenAI setup
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not set! Export it before running this script.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-3-small"

# -----------------------------------------------------------------------------
# Database config (matches your .env: 127.0.0.1, mh_catalog, app_user/app_user)
# -----------------------------------------------------------------------------
DB_CONFIG: Dict[str, object] = {
    "host": os.getenv("POSTGRES_HOST", "127.0.0.1"),
    "database": os.getenv("POSTGRES_DB", "mh_catalog"),
    "user": os.getenv("POSTGRES_USER", "app_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "app_user"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_db_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        sys.exit(1)


def create_embedding(text: str) -> Optional[List[float]]:
    """Create OpenAI embedding for given text."""
    try:
        # Truncate to be safe
        text = text[:8000]
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None


def serialize_content(raw_content) -> str:
    """
    maryland_mental_health_data.content is JSONB.
    In Python that becomes dict/list, so we must convert to string
    before slicing/embedding.
    """
    if raw_content is None:
        return ""

    # If already a dict or list, dump to JSON string
    if isinstance(raw_content, (dict, list)):
        return json.dumps(raw_content, ensure_ascii=False)

    # Fallback: just string it
    return str(raw_content)


# -----------------------------------------------------------------------------
# Main indexing logic
# -----------------------------------------------------------------------------
def index_maryland_records():
    """Index all maryland_mental_health_data records into the chunks table."""
    conn = get_db_connection()
    cur = conn.cursor()

    # How many records do we have?
    cur.execute("SELECT COUNT(*) FROM maryland_mental_health_data;")
    total = cur.fetchone()[0]
    logger.info(f"Found {total} records to index")

    if total == 0:
        logger.error("No records found! Run index_maryland_mental_health_intelligent.py first.")
        cur.close()
        conn.close()
        return

    # Fetch all rows (for now; can batch later if needed)
    cur.execute(
        """
        SELECT id, source, dataset_id, title, content, location, metadata, url
        FROM maryland_mental_health_data
        ORDER BY id;
        """
    )
    rows = cur.fetchall()

    indexed = 0
    skipped = 0

    for idx, row in enumerate(rows, start=1):
        record_id, source, dataset_id, title, content, location, metadata, url = row

        # Convert JSONB content → string before slicing
        content_str = serialize_content(content)
        short_content = content_str[:2000]

        # Build chunk text
        chunk_content = f"""Source: {source}
Title: {title or 'N/A'}
Location: {location or 'Maryland'}
Dataset ID: {dataset_id or 'N/A'}

Content:
{short_content}
""".strip()

        logger.info(f"Creating embedding for record {record_id} ({idx}/{total})")
        embedding = create_embedding(chunk_content)

        if not embedding:
            skipped += 1
            continue

        # pgvector accepts something like '[0.1,0.2,...]'
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

        # Chunk id + dataset uid
        chunk_id = f"MD_{source}_{record_id}"
        dataset_uid = f"maryland_data_{source}_{dataset_id or record_id}"

        try:
            cur.execute(
                """
                INSERT INTO chunks
                    (dataset_uid, org, chunk_id, content, metadata, embedding)
                VALUES
                    (%s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (chunk_id) DO NOTHING;
                """,
                (
                    dataset_uid,
                    source,  # treat 'CDC', 'Maryland Open Data', 'PubMed' as org
                    chunk_id,
                    chunk_content,
                    Json(
                        {
                            "source": source,
                            "title": title,
                            "location": location,
                            "original_id": record_id,
                            "url": url,
                        }
                    ),
                    embedding_str,
                ),
            )

            indexed += 1

            # Commit every 100 new chunks
            if indexed % 100 == 0:
                conn.commit()
                logger.info(
                    f"Progress: {indexed}/{total} indexed, {skipped} skipped"
                )
                # very light rate limit
                time.sleep(0.5)

        except Exception as e:
            logger.error(f"Insert error for record {record_id}: {e}")
            skipped += 1
            conn.rollback()  # rollback this failed insert and continue

    # Final commit
    conn.commit()
    cur.close()
    conn.close()

    logger.info(
        f"""
✅ Indexing complete!
- Total records in maryland_mental_health_data: {total}
- Indexed into chunks: {indexed}
- Skipped (errors / embedding failures): {skipped}
"""
    )


if __name__ == "__main__":
    index_maryland_records()
