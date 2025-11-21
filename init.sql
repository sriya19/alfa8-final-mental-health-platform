-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Datasets table (metadata catalog)
CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    uid TEXT NOT NULL,
    org TEXT,
    title TEXT,
    name TEXT,
    source TEXT,
    location TEXT,
    link TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(uid, org)
);

CREATE INDEX idx_datasets_org ON datasets(org);
CREATE INDEX idx_datasets_uid ON datasets(uid);

-- Ingested datasets table (MinIO tracking)
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
);

CREATE INDEX idx_ingested_org_uid ON ingested_datasets(org, dataset_uid);

-- Chunks table (RAG embeddings)
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

CREATE INDEX idx_chunks_dataset ON chunks(org, dataset_uid);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);