#!/usr/bin/env bash
set -euo pipefail

echo "== Step 1: datasets =="
docker exec -i pg psql -U app_user -d mh_catalog <<'SQL'
-- Ensure table
CREATE TABLE IF NOT EXISTS datasets (
  id BIGINT,
  name TEXT NOT NULL,
  source_org TEXT NOT NULL,
  source_url TEXT,
  description TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (source_org, source_url)
);

-- Add id if missing
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='datasets' AND column_name='id'
  ) THEN
    ALTER TABLE datasets ADD COLUMN id BIGINT;
  END IF;
END$$;

-- Create/attach sequence and PK
CREATE SEQUENCE IF NOT EXISTS datasets_id_seq;
ALTER SEQUENCE datasets_id_seq OWNED BY datasets.id;
ALTER TABLE datasets ALTER COLUMN id SET DEFAULT nextval('datasets_id_seq');
UPDATE datasets SET id = nextval('datasets_id_seq') WHERE id IS NULL;
ALTER TABLE datasets ALTER COLUMN id SET NOT NULL;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid='public.datasets'::regclass AND contype='p'
  ) THEN
    ALTER TABLE datasets ADD CONSTRAINT datasets_pkey PRIMARY KEY (id);
  END IF;
END$$;

CREATE UNIQUE INDEX IF NOT EXISTS datasets_org_url_uniq
  ON datasets (source_org, COALESCE(source_url,''));
CREATE INDEX IF NOT EXISTS idx_datasets_org ON datasets(source_org);
CREATE INDEX IF NOT EXISTS idx_datasets_created ON datasets(created_at);
SQL

echo "== Step 2: pgvector & semantic_index =="
docker exec -i pg psql -U app_user -d mh_catalog <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS semantic_index (
  id BIGSERIAL PRIMARY KEY,
  org TEXT NOT NULL,
  uid TEXT NOT NULL,
  name TEXT,
  description TEXT,
  embedding vector(1536),
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(org, uid)
);

CREATE INDEX IF NOT EXISTS semantic_index_org_uid_idx ON semantic_index(org, uid);
CREATE INDEX IF NOT EXISTS semantic_index_embedding_idx
  ON semantic_index USING ivfflat (embedding vector_cosine_ops);
SQL

echo "== Step 3: data_chunks (RAG over real data) =="
docker exec -i pg psql -U app_user -d mh_catalog <<'SQL'
CREATE TABLE IF NOT EXISTS data_chunks (
  id BIGSERIAL PRIMARY KEY,
  org TEXT NOT NULL,
  dataset_uid TEXT NOT NULL,
  chunk_id TEXT UNIQUE NOT NULL,
  content TEXT NOT NULL,
  summary TEXT,
  metadata JSONB,
  embedding vector(1536),
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS data_chunks_embedding_idx
  ON data_chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS data_chunks_org_uid_idx
  ON data_chunks(org, dataset_uid);
SQL

echo "== Step 4: legacy chunks (optional) =="
docker exec -i pg psql -U app_user -d mh_catalog <<'SQL'
CREATE TABLE IF NOT EXISTS chunks (
  id SERIAL PRIMARY KEY,
  org TEXT,
  dataset_uid TEXT,
  chunk_id TEXT UNIQUE,
  content TEXT,
  embedding vector(1536),
  created_at TIMESTAMPTZ DEFAULT now()
);
SQL

echo "== Step 5: Verify =="
docker exec -i pg psql -U app_user -d mh_catalog -c "\dt"
