"""
Data Ingestion Module - Socrata to MinIO to PostgreSQL Pipeline
Fixed: Uses ONLY mh-raw bucket, proper error handling, auto-index support
"""

import os
import io
import json
import pandas as pd
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
import boto3
from botocore.exceptions import ClientError
import hashlib
import traceback

logger = logging.getLogger(__name__)

# MinIO/S3 Configuration - SINGLE BUCKET ONLY
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio-1:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "mh-raw")  # Single bucket for all data
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Socrata domains
DOMAINS = {
    "CDC": "data.cdc.gov",
    "SAMHSA": "data.samhsa.gov",
    "Maryland": "opendata.maryland.gov"
}

def get_s3_client():
    """
    Get configured S3/MinIO client
    
    Returns:
        boto3.client: Configured S3 client
        
    Raises:
        Exception: If client creation fails
    """
    try:
        client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name=AWS_REGION,
            use_ssl=False,
            verify=False
        )
        logger.info(f"S3 client created for endpoint: {S3_ENDPOINT}")
        return client
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        raise

def ensure_bucket_exists():
    """
    Ensure ONLY the mh-raw bucket exists, create if not
    
    Returns:
        bool: True if bucket exists or was created successfully
        
    Raises:
        Exception: If bucket operations fail critically
    """
    s3 = get_s3_client()
    
    try:
        # List all buckets
        response = s3.list_buckets()
        existing_buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
        logger.info(f"Existing buckets: {existing_buckets}")
        
        # Check if mh-raw exists
        if S3_BUCKET not in existing_buckets:
            logger.info(f"Creating bucket: {S3_BUCKET}")
            s3.create_bucket(Bucket=S3_BUCKET)
            logger.info(f"✅ Created bucket: {S3_BUCKET}")
        else:
            logger.info(f"✅ Bucket already exists: {S3_BUCKET}")
            
        # Verify we can write to it
        test_key = "_test_write.txt"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=test_key,
            Body=b"test",
            ContentType='text/plain'
        )
        s3.delete_object(Bucket=S3_BUCKET, Key=test_key)
        logger.info(f"✅ Bucket {S3_BUCKET} is writable")
        
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"Bucket operation failed with error code {error_code}: {e}")
        
        if error_code == 'NoSuchBucket':
            # Try to create it
            try:
                s3.create_bucket(Bucket=S3_BUCKET)
                logger.info(f"✅ Created bucket after NoSuchBucket error: {S3_BUCKET}")
                return True
            except Exception as create_error:
                logger.error(f"Failed to create bucket: {create_error}")
                raise
        raise
    except Exception as e:
        logger.error(f"Unexpected error ensuring bucket exists: {e}")
        raise

def fetch_dataset_from_socrata(org: str, dataset_uid: str, limit: int = 5000) -> pd.DataFrame:
    """
    Fetch dataset from Socrata API
    
    Args:
        org: Organization name (CDC, SAMHSA, Maryland)
        dataset_uid: The 4x4 dataset identifier (e.g., "abc1-def2")
        limit: Maximum number of rows to fetch (default: 5000)
        
    Returns:
        pd.DataFrame: Dataset as a pandas DataFrame
        
    Raises:
        ValueError: If organization is unknown or dataset not found
        requests.exceptions.RequestException: If API request fails
    """
    domain = DOMAINS.get(org)
    if not domain:
        raise ValueError(f"Unknown organization: {org}. Valid options: {list(DOMAINS.keys())}")
    
    # Construct Socrata API URL
    url = f"https://{domain}/resource/{dataset_uid}.json"
    params = {"$limit": limit}
    
    # Add Socrata app token if available
    app_token = os.getenv("SOCRATA_APP_TOKEN")
    headers = {"Accept": "application/json"}
    if app_token:
        headers["X-App-Token"] = app_token
    
    try:
        logger.info(f"Fetching dataset from: {url}")
        logger.info(f"Parameters: {params}")
        
        response = requests.get(url, params=params, headers=headers, timeout=60)
        
        if response.status_code == 404:
            logger.error(f"Dataset not found: {dataset_uid} on {domain}")
            raise ValueError(f"Dataset {dataset_uid} not found on {domain}")
        
        response.raise_for_status()
        
        data = response.json()
        if not data:
            logger.warning(f"No data returned for dataset {dataset_uid}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        logger.info(f"✅ Fetched {len(df)} rows from {dataset_uid}")
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch dataset {dataset_uid}: {e}")
        logger.error(f"Full error: {traceback.format_exc()}")
        raise

def store_dataset_in_minio(org: str, dataset_uid: str, df: pd.DataFrame) -> str:
    """
    Store dataset in MinIO as Parquet file in mh-raw bucket ONLY
    
    Args:
        org: Organization name
        dataset_uid: Dataset identifier
        df: DataFrame to store
        
    Returns:
        str: The S3 key where the dataset was stored
        
    Raises:
        Exception: If storage operation fails
    """
    # Ensure bucket exists first
    ensure_bucket_exists()
    
    # Create S3 key with organization structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_key = f"{org}/{dataset_uid}/{timestamp}.parquet"
    
    try:
        # Convert DataFrame to Parquet in memory
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, engine='pyarrow')
        buffer.seek(0)
        
        # Upload to MinIO
        s3 = get_s3_client()
        
        logger.info(f"Uploading to MinIO: bucket={S3_BUCKET}, key={s3_key}")
        
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=buffer.getvalue(),
            ContentType='application/octet-stream',
            Metadata={
                'org': org,
                'dataset_uid': dataset_uid,
                'rows': str(len(df)),
                'columns': str(len(df.columns)),
                'ingested_at': timestamp
            }
        )
        
        # Verify the upload
        response = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
        file_size = response['ContentLength']
        
        logger.info(f"✅ Stored dataset in MinIO: s3://{S3_BUCKET}/{s3_key} (size: {file_size} bytes)")
        return s3_key
        
    except Exception as e:
        logger.error(f"Failed to store dataset in MinIO: {e}")
        logger.error(f"Full error: {traceback.format_exc()}")
        raise

def store_metadata_in_postgres(org: str, dataset_uid: str, df: pd.DataFrame, s3_key: str) -> Dict[str, Any]:
    """
    Store dataset metadata in PostgreSQL
    
    Args:
        org: Organization name
        dataset_uid: Dataset identifier
        df: DataFrame (for metadata extraction)
        s3_key: S3 key where data is stored
        
    Returns:
        dict: Metadata including dataset_id and already_indexed status
        
    Raises:
        Exception: If database operation fails
    """
    import psycopg2
    from psycopg2.extras import Json
    
    # Database configuration
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "pg"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "database": os.getenv("POSTGRES_DB", "mh_catalog"),
        "user": os.getenv("POSTGRES_USER", "app_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "app_user")
    }
    
    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingested_datasets (
                id SERIAL PRIMARY KEY,
                org TEXT NOT NULL,
                dataset_uid TEXT NOT NULL,
                dataset_name TEXT,
                s3_key TEXT NOT NULL,
                s3_bucket TEXT DEFAULT 'mh-raw',
                row_count INTEGER,
                column_count INTEGER,
                columns_list TEXT,
                data_hash TEXT,
                ingested_at TIMESTAMP DEFAULT NOW(),
                indexed BOOLEAN DEFAULT FALSE,
                UNIQUE(org, dataset_uid)
            )
        """)
        
        # Calculate data hash for deduplication
        data_hash = hashlib.md5(df.to_json().encode()).hexdigest()
        
        # Get dataset metadata
        columns_list = json.dumps(df.columns.tolist())
        
        # Insert or update metadata
        cursor.execute("""
            INSERT INTO ingested_datasets 
            (org, dataset_uid, s3_key, s3_bucket, row_count, column_count, columns_list, data_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (org, dataset_uid) 
            DO UPDATE SET 
                s3_key = EXCLUDED.s3_key,
                s3_bucket = EXCLUDED.s3_bucket,
                row_count = EXCLUDED.row_count,
                column_count = EXCLUDED.column_count,
                columns_list = EXCLUDED.columns_list,
                data_hash = EXCLUDED.data_hash,
                ingested_at = NOW()
            RETURNING id, indexed
        """, (org, dataset_uid, s3_key, S3_BUCKET, len(df), len(df.columns), columns_list, data_hash))
        
        result = cursor.fetchone()
        dataset_id = result[0]
        already_indexed = result[1]
        
        conn.commit()
        
        logger.info(f"✅ Stored metadata in PostgreSQL for dataset {dataset_uid} (id: {dataset_id})")
        
        return {
            "dataset_id": dataset_id,
            "already_indexed": already_indexed
        }
        
    except Exception as e:
        logger.error(f"Failed to store metadata in PostgreSQL: {e}")
        logger.error(f"Full error: {traceback.format_exc()}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            cursor.close()
            conn.close()

def ingest_dataset(org: str, dataset_uid: str, auto_index: bool = False) -> Dict[str, Any]:
    """
    Complete ingestion pipeline: Socrata → MinIO (mh-raw bucket) → PostgreSQL → (optionally) RAG indexing
    
    Args:
        org: Organization name (CDC, SAMHSA, Maryland)
        dataset_uid: The 4x4 dataset identifier
        auto_index: If True, automatically index for RAG after ingestion
        
    Returns:
        dict: Ingestion result with status, metrics, and any errors
        
    Example:
        result = ingest_dataset("CDC", "abc1-def2", auto_index=True)
        if result["success"]:
            print(f"Ingested {result['rows']} rows")
            if result["indexed"]:
                print(f"Created {result['chunks_created']} chunks")
    """
    try:
        logger.info(f"Starting ingestion for {org}/{dataset_uid}")
        logger.info(f"Configuration: S3_ENDPOINT={S3_ENDPOINT}, S3_BUCKET={S3_BUCKET}")
        
        # Step 1: Fetch from Socrata
        try:
            df = fetch_dataset_from_socrata(org, dataset_uid)
        except Exception as e:
            error_msg = f"Failed to fetch dataset: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "ingested": False,
                "error": error_msg,
                "reason": error_msg
            }
        
        if df.empty:
            return {
                "success": False,
                "ingested": False,
                "reason": "No data available for this dataset"
            }
        
        # Step 2: Store in MinIO (mh-raw bucket)
        try:
            s3_key = store_dataset_in_minio(org, dataset_uid, df)
        except Exception as e:
            error_msg = f"Failed to store in MinIO: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "ingested": False,
                "error": error_msg,
                "reason": error_msg
            }
        
        # Step 3: Store metadata in PostgreSQL
        try:
            metadata = store_metadata_in_postgres(org, dataset_uid, df, s3_key)
        except Exception as e:
            error_msg = f"Failed to store metadata: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "ingested": False,
                "error": error_msg,
                "reason": error_msg
            }
        
        # Step 4: Auto-index if requested
        indexed = False
        chunks_created = 0
        
        if auto_index and not metadata.get("already_indexed"):
            try:
                from app.rag import index_dataset_for_rag
                index_result = index_dataset_for_rag(org, dataset_uid, s3_key)
                indexed = index_result.get("success", False)
                chunks_created = index_result.get("chunks_created", 0)
                logger.info(f"Auto-indexing result: indexed={indexed}, chunks={chunks_created}")
            except Exception as e:
                logger.warning(f"Auto-indexing failed (non-fatal): {e}")
                # Don't fail the whole ingestion if indexing fails
        
        logger.info(f"✅ Successfully ingested {org}/{dataset_uid}: {len(df)} rows")
        
        return {
            "success": True,
            "ingested": True,
            "rows": len(df),
            "columns": len(df.columns),
            "s3_key": s3_key,
            "s3_bucket": S3_BUCKET,
            "indexed": indexed,
            "chunks_created": chunks_created,
            "dataset_id": metadata.get("dataset_id")
        }
        
    except Exception as e:
        error_msg = f"Ingestion failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full error: {traceback.format_exc()}")
        return {
            "success": False,
            "ingested": False,
            "error": error_msg,
            "reason": error_msg
        }

def get_dataset_from_minio(org: str, dataset_uid: str) -> pd.DataFrame:
    """
    Retrieve dataset from MinIO (mh-raw bucket)
    
    Args:
        org: Organization name
        dataset_uid: Dataset identifier
        
    Returns:
        pd.DataFrame: Dataset as a pandas DataFrame, empty DataFrame if not found
        
    Example:
        df = get_dataset_from_minio("CDC", "abc1-def2")
        if not df.empty:
            print(f"Loaded {len(df)} rows")
    """
    import psycopg2
    
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "pg"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "database": os.getenv("POSTGRES_DB", "mh_catalog"),
        "user": os.getenv("POSTGRES_USER", "app_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "app_user")
    }
    
    try:
        # Get S3 key from database
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT s3_key 
            FROM ingested_datasets 
            WHERE org = %s AND dataset_uid = %s
            ORDER BY ingested_at DESC
            LIMIT 1
        """, (org, dataset_uid))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            logger.warning(f"No ingested dataset found for {org}/{dataset_uid}")
            return pd.DataFrame()
        
        s3_key = result[0]
        
        # Retrieve from MinIO
        s3 = get_s3_client()
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        
        # Read Parquet from S3
        df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
        logger.info(f"✅ Retrieved dataset from MinIO: {s3_key} ({len(df)} rows)")
        return df
        
    except Exception as e:
        logger.error(f"Failed to retrieve dataset from MinIO: {e}")
        return pd.DataFrame()

def list_ingested_datasets(org: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all ingested datasets, optionally filtered by organization
    
    Args:
        org: Optional organization filter
        
    Returns:
        list: List of dataset metadata dictionaries
        
    Example:
        datasets = list_ingested_datasets(org="CDC")
        for ds in datasets:
            print(f"{ds['dataset_uid']}: {ds['row_count']} rows")
    """
    import psycopg2
    from psycopg2.extras import RealDictCursor
    
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "pg"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "database": os.getenv("POSTGRES_DB", "mh_catalog"),
        "user": os.getenv("POSTGRES_USER", "app_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "app_user")
    }
    
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        if org:
            cursor.execute("""
                SELECT org, dataset_uid, dataset_name, row_count, column_count, 
                       indexed, ingested_at, s3_key
                FROM ingested_datasets
                WHERE org = %s
                ORDER BY ingested_at DESC
            """, (org,))
        else:
            cursor.execute("""
                SELECT org, dataset_uid, dataset_name, row_count, column_count, 
                       indexed, ingested_at, s3_key
                FROM ingested_datasets
                ORDER BY ingested_at DESC
            """)
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error(f"Failed to list ingested datasets: {e}")
        return []

def delete_ingested_dataset(org: str, dataset_uid: str, delete_from_minio: bool = True) -> Dict[str, Any]:
    """
    Delete an ingested dataset from PostgreSQL and optionally from MinIO
    
    Args:
        org: Organization name
        dataset_uid: Dataset identifier
        delete_from_minio: If True, also delete from MinIO (default: True)
        
    Returns:
        dict: Deletion result with success status
        
    Example:
        result = delete_ingested_dataset("CDC", "abc1-def2")
        if result["success"]:
            print("Dataset deleted")
    """
    import psycopg2
    
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "pg"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "database": os.getenv("POSTGRES_DB", "mh_catalog"),
        "user": os.getenv("POSTGRES_USER", "app_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "app_user")
    }
    
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Get S3 key first if we need to delete from MinIO
        if delete_from_minio:
            cursor.execute("""
                SELECT s3_key 
                FROM ingested_datasets 
                WHERE org = %s AND dataset_uid = %s
            """, (org, dataset_uid))
            
            result = cursor.fetchone()
            if result:
                s3_key = result[0]
                try:
                    s3 = get_s3_client()
                    s3.delete_object(Bucket=S3_BUCKET, Key=s3_key)
                    logger.info(f"✅ Deleted from MinIO: {s3_key}")
                except Exception as e:
                    logger.warning(f"Failed to delete from MinIO: {e}")
        
        # Delete from database
        cursor.execute("""
            DELETE FROM ingested_datasets 
            WHERE org = %s AND dataset_uid = %s
        """, (org, dataset_uid))
        
        rows_deleted = cursor.rowcount
        conn.commit()
        
        cursor.close()
        conn.close()
        
        if rows_deleted > 0:
            logger.info(f"✅ Deleted dataset {org}/{dataset_uid} from database")
            return {"success": True, "message": "Dataset deleted"}
        else:
            return {"success": False, "message": "Dataset not found"}
        
    except Exception as e:
        logger.error(f"Failed to delete dataset: {e}")
        return {"success": False, "error": str(e)}


# Utility function for testing
def test_ingestion():
    """Test function to verify ingestion pipeline"""
    print("Testing ingestion pipeline...")
    
    # Test with a small CDC dataset
    test_org = "CDC"
    test_uid = "hyst-znpv"  # Small test dataset
    
    print(f"\n1. Testing ingestion: {test_org}/{test_uid}")
    result = ingest_dataset(test_org, test_uid, auto_index=False)
    
    if result["success"]:
        print(f"✅ Ingestion successful!")
        print(f"   Rows: {result['rows']}")
        print(f"   Columns: {result['columns']}")
        print(f"   S3 Key: {result['s3_key']}")
    else:
        print(f"❌ Ingestion failed: {result.get('reason', 'Unknown error')}")
    
    print("\n2. Testing retrieval from MinIO")
    df = get_dataset_from_minio(test_org, test_uid)
    if not df.empty:
        print(f"✅ Retrieved {len(df)} rows from MinIO")
    else:
        print("❌ Failed to retrieve from MinIO")
    
    print("\n✅ Ingestion tests complete!")


if __name__ == "__main__":
    # Run tests if executed directly
    test_ingestion()