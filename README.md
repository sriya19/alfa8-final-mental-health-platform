# Mental Health Platform

A comprehensive mental health data platform that enables discovery, ingestion, and intelligent querying of mental health datasets from CDC and SAMHSA through a RAG (Retrieval-Augmented Generation) powered interface.

## Overview

This platform provides a unified interface for working with mental health data from government sources. It combines:
- **Data Discovery & Ingestion**: Search and ingest datasets from CDC and SAMHSA Socrata catalogs
- **Vector-Based RAG**: Semantic search over dataset metadata and content using OpenAI embeddings
- **AI-Powered Insights**: Query datasets using natural language with LLM-powered interpretation
- **Interactive UI**: Streamlit-based web interface for exploration and visualization
- **Scalable Storage**: MinIO (S3-compatible) for Parquet data storage, PostgreSQL with pgvector for metadata

## Architecture

### Components

1. **Backend API (FastAPI)**
   - RESTful API for data operations
   - Located in: `backend/app/`
   - Endpoints for catalog search, dataset ingestion, RAG queries, and data preview
   - Modules:
     - `main.py`: Core API endpoints and request handlers
     - `socrata.py`: Socrata API integration for CDC/SAMHSA data
     - `rag.py`: RAG implementation with vector similarity search
     - `semantic.py`: Semantic search over dataset metadata
     - `answer.py`: LLM-powered question answering
     - `interpret.py`: Natural language query interpretation
     - `ingest.py`: Data ingestion and S3 storage
     - `db.py`: Database connection and ORM setup
     - `llm.py`: OpenAI API integration
     - `config.py`: Configuration management

2. **Database (PostgreSQL + pgvector)**
   - Stores dataset metadata and vector embeddings
   - Tables:
     - `datasets`: Dataset registry with source information
     - `chunks`: Vector-indexed content chunks for RAG (1536-dim embeddings)
     - `meta_docs`: Document metadata with embeddings
   - Uses pgvector extension for efficient similarity search

3. **Object Storage (MinIO)**
   - S3-compatible storage for Parquet files
   - Stores ingested datasets in `raw/{org}/{uid}.parquet` format
   - Web console available on port 9001

4. **UI Application (Streamlit)**
   - Interactive web interface
   - Located in: `ui_app/streamlit_app.py`
   - Features:
     - Dataset search and discovery
     - Quick preview and visualization
     - RAG-powered question answering
     - Dataset management and indexing status

5. **Indexing Script**
   - `index_cdc_mental_health_intelligent.py`: Automated discovery and indexing of CDC mental health datasets
   - Uses LLM to generate comprehensive search queries
   - Targets 5000+ mental health-related datasets

## Database Schema

### PostgreSQL (Port 5432)
- **Database**: `mh_catalog`
- **Extensions**: pgvector
- **Key Tables**:
  - `datasets`: Source dataset metadata
  - `chunks`: RAG vector embeddings (1536-dim OpenAI embeddings)
  - `meta_docs`: Document metadata
  - `semantic_index`: Semantic search index (created on first use)
  - `data_chunks`: Indexed dataset content (created on first use)

### MinIO Object Storage (Ports 9000, 9001)
- **Bucket**: `mh-raw`
- **Format**: Apache Parquet
- **Structure**: `raw/{org}/{dataset_uid}.parquet`

## Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine + Docker Compose (Ubuntu)
- 4GB+ RAM recommended
- 10GB+ disk space for data storage
- OpenAI API key (for RAG and LLM features)

## Installation
Please note that - index_cdc_mental_health_intelligent.py will take long to run. So you can skip that step while running and run the remaning 2 indexing scripts.
### Windows

1. **Install Docker Desktop**
   - Download from: https://www.docker.com/products/docker-desktop
   - Run the installer and follow the setup wizard
   - Ensure WSL 2 is enabled (Docker Desktop will prompt if needed)
   - Start Docker Desktop and wait for it to fully initialize

2. **Clone or Extract the Repository**
   ```powershell
   cd C:\Users\YourName\Projects
   # If you have the folder, navigate to it
   cd mental-health-platform-master
   ```

3. **Create Environment File**
   ```powershell
   copy .env.example .env
   notepad .env
   ```

   Update the following required settings:
   ```env
   # PostgreSQL Configuration
   POSTGRES_HOST=postgres
   POSTGRES_PORT=5432
   POSTGRES_DB=mh_catalog
   POSTGRES_USER=app_user
   POSTGRES_PASSWORD=YourSecurePassword123

   # MinIO Configuration
   MINIO_ROOT_USER=minioadmin
   MINIO_ROOT_PASSWORD=YourSecurePassword456
   S3_BUCKET=mh-raw
   S3_ENDPOINT=http://minio:9000
   S3_ACCESS_KEY=minioadmin
   S3_SECRET_KEY=YourSecurePassword456

   # OpenAI Configuration (Required for RAG features)
   OPENAI_API_KEY=sk-your-key-here
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_BASE_URL=https://api.openai.com/v1

   # Optional: Socrata API token for higher rate limits
   SOCRATA_APP_TOKEN=your-token-here
   ```

4. **Start the Platform**
   ```powershell
   docker-compose up -d
   ```

5. **Verify Installation**
   ```powershell
   docker-compose ps
   ```

   All services should show "Up" status:
   - `postgres`: Database
   - `minio`: Object storage
   - `backend`: API server
   - `create-bucket`: Bucket initialization (will show "Up" or "Completed")

6. **Access the Applications**
   - Backend API: http://localhost:8000/docs
   - MinIO Console: http://localhost:9001 (login with MINIO_ROOT_USER/PASSWORD)
   - Health Check: http://localhost:8000/health

### macOS

1. **Install Docker Desktop**
   ```bash
   # Download from https://www.docker.com/products/docker-desktop
   # Or install via Homebrew:
   brew install --cask docker
   ```

   - Open Docker Desktop from Applications
   - Wait for Docker to start (whale icon in menu bar)

2. **Clone or Navigate to Repository**
   ```bash
   cd ~/Projects
   cd mental-health-platform-master
   ```

3. **Create Environment File**
   ```bash
   cp .env.example .env
   nano .env  # or use your preferred editor
   ```

   Update the configuration as shown in the Windows section above.

4. **Start the Platform**
   ```bash
   docker-compose up -d
   ```

5. **Verify Installation**
   ```bash
   docker-compose ps
   docker-compose logs backend
   ```

6. **Access the Applications**
   - Backend API: http://localhost:8000/docs
   - MinIO Console: http://localhost:9001
   - Health Check: http://localhost:8000/health

### Ubuntu Linux

1. **Install Docker Engine and Docker Compose**
   ```bash
   # Update package index
   sudo apt-get update

   # Install prerequisites
   sudo apt-get install -y ca-certificates curl gnupg lsb-release

   # Add Docker's official GPG key
   sudo mkdir -p /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

   # Set up the repository
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

   # Install Docker Engine
   sudo apt-get update
   sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

   # Add your user to docker group (to run without sudo)
   sudo usermod -aG docker $USER
   newgrp docker  # Or log out and back in

   # Verify installation
   docker --version
   docker compose version
   ```

2. **Navigate to Repository**
   ```bash
   cd ~/projects/mental-health-platform-master
   ```

3. **Create Environment File**
   ```bash
   cp .env.example .env
   nano .env  # or vim, emacs, etc.
   ```

   Update the configuration as shown in the Windows section above.

4. **Start the Platform**
   ```bash
   docker compose up -d
   ```

5. **Verify Installation**
   ```bash
   docker compose ps
   docker compose logs backend
   ```

6. **Access the Applications**
   - Backend API: http://localhost:8000/docs
   - MinIO Console: http://localhost:9001
   - Health Check: http://localhost:8000/health

## Running the UI Application

The Streamlit UI is not dockerized by default. To run it:

1. **Install Python Dependencies**
   ```bash
   cd ui_app
   pip install streamlit requests pandas numpy plotly pydeck
   ```

2. **Configure Backend URL** (if needed)
   ```bash
   # Create .env in ui_app directory
   echo "BACKEND_URL=http://localhost:8000" > .env
   ```

3. **Start Streamlit**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access UI**
   - Open: http://localhost:8501

## Usage

### 1. Search and Ingest Datasets

**Via API:**
```bash
# Search CDC catalog
curl "http://localhost:8000/catalog/search?org=CDC&q=mental%20health"

# Ingest a dataset with auto-indexing for RAG
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "org": "CDC",
    "query": "youth mental health",
    "auto_index": true
  }'
```

**Via UI:**
1. Enter search query (e.g., "depression", "substance abuse")
2. Select organization (CDC or SAMHSA)
3. Click "Search Catalog"
4. Review results and click "Ingest" to store datasets

### 2. Query with RAG

**Via API:**
```bash
# Ask a question about ingested data
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the trends in youth depression?",
    "org": "CDC"
  }'
```

**Via UI:**
1. Navigate to "Ask Questions" tab
2. Enter your question
3. View AI-generated answer with source citations

### 3. Index Datasets for RAG

```bash
# Index a specific dataset
curl -X POST http://localhost:8000/index_dataset \
  -H "Content-Type: application/json" \
  -d '{
    "org": "CDC",
    "uid": "dataset-uid-here",
    "limit_rows": 5000
  }'

# Batch index all datasets for an organization
curl -X POST "http://localhost:8000/batch_index?org=CDC"

# Check indexing status
curl "http://localhost:8000/rag_status?org=CDC"
```

### 4. Automated Dataset Discovery

Run the intelligent indexer to automatically discover and ingest mental health datasets:

```bash
# Ensure OPENAI_API_KEY is set in environment
export OPENAI_API_KEY=sk-your-key-here

# Run the indexer
python index_cdc_mental_health_intelligent.py
```

This script:
- Uses LLM to generate 100+ diverse mental health search queries
- Searches CDC catalog for each query
- Ingests unique datasets automatically
- Targets 5000+ mental health datasets

## API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `GET /catalog/search` - Search Socrata catalog
- `POST /ingest` - Ingest dataset from Socrata
- `GET /datasets` - List ingested datasets
- `GET /datasets/preview` - Preview dataset from storage
- `GET /datasets/quick_preview` - Live preview from Socrata

### RAG & Search
- `POST /index_dataset` - Index dataset for RAG
- `POST /batch_index` - Index all datasets for an org
- `GET /rag_status` - Check indexing status
- `DELETE /datasets/{uid}/index` - Remove dataset index
- `POST /semantic/search` - Semantic search over metadata
- `POST /answer/query` - Answer questions using RAG

### Statistics
- `GET /stats` - System-wide statistics

Full API documentation: http://localhost:8000/docs

## Management Commands

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f postgres
```

### Restart Services
```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart backend
```

### Stop Platform
```bash
docker-compose down
```

### Stop and Remove Data
```bash
docker-compose down -v  # WARNING: Deletes all data!
```

### Database Access
```bash
# Connect to PostgreSQL
docker exec -it pg psql -U app_user -d mh_catalog

# Example queries:
# \dt                          -- List tables
# SELECT * FROM datasets;      -- View datasets
# SELECT COUNT(*) FROM chunks; -- Count indexed chunks
```

### MinIO Access
1. Open http://localhost:9001
2. Login with MINIO_ROOT_USER/MINIO_ROOT_PASSWORD
3. Browse the `mh-raw` bucket
4. View/download Parquet files

## Troubleshooting

### Backend Won't Start
```bash
# Check logs
docker-compose logs backend

# Common issues:
# 1. Database not ready - wait 30 seconds and check again
# 2. Missing .env file - ensure .env exists with all required variables
# 3. Port 8000 in use - stop other services using this port
```

### Database Connection Errors
```bash
# Verify PostgreSQL is running
docker-compose ps postgres

# Check if database is accepting connections
docker exec pg pg_isready -U app_user

# Restart database
docker-compose restart postgres
```

### MinIO Connection Errors
```bash
# Verify MinIO is running
docker-compose ps minio

# Check bucket creation
docker-compose logs create-bucket

# Restart MinIO
docker-compose restart minio create-bucket
```

### RAG Not Working
1. Ensure OPENAI_API_KEY is set in .env
2. Check if datasets are indexed: `curl http://localhost:8000/rag_status?org=CDC`
3. Index datasets if needed: `curl -X POST http://localhost:8000/batch_index?org=CDC`
4. Verify OpenAI API quota and billing

### Port Conflicts
If ports 8000, 9000, 9001, or 5432 are already in use:

Edit `docker-compose.yml` to change port mappings:
```yaml
services:
  backend:
    ports:
      - "8001:8000"  # Change host port (left side) only
```

### Out of Memory
If containers are being killed:
- Increase Docker memory limit (Docker Desktop → Settings → Resources)
- Recommended: 4GB minimum, 8GB for large datasets

## Configuration Reference

### Environment Variables

#### Required
- `POSTGRES_USER` - Database username
- `POSTGRES_PASSWORD` - Database password
- `POSTGRES_DB` - Database name
- `MINIO_ROOT_USER` - MinIO admin username
- `MINIO_ROOT_PASSWORD` - MinIO admin password
- `OPENAI_API_KEY` - OpenAI API key (for RAG features)

#### Optional
- `SOCRATA_APP_TOKEN` - Socrata API token for higher rate limits
- `OPENAI_MODEL` - LLM model (default: gpt-4o-mini)
- `OPENAI_BASE_URL` - OpenAI API base URL
- `S3_BUCKET` - MinIO bucket name (default: mh-raw)
- `TZ` - Timezone (default: America/New_York)

### Data Sources

- **CDC**: https://data.cdc.gov
- **SAMHSA**: https://data.samhsa.gov

Both use the Socrata Open Data API (SODA).

## Development

### Project Structure
```
mental-health-platform-master/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py         # API endpoints
│   │   ├── rag.py          # RAG implementation
│   │   ├── semantic.py     # Semantic search
│   │   ├── answer.py       # Q&A module
│   │   ├── socrata.py      # Socrata API client
│   │   ├── ingest.py       # Data ingestion
│   │   ├── db.py           # Database connection
│   │   ├── llm.py          # LLM integration
│   │   └── config.py       # Configuration
│   ├── Dockerfile
│   └── requirements.txt
├── ui_app/
│   └── streamlit_app.py    # Streamlit UI
├── infra/
│   └── init.sql            # Database schema
├── docker-compose.yml      # Docker orchestration
├── .env.example            # Environment template
├── index_cdc_mental_health_intelligent.py  # Auto-indexer
└── README.md
```

### Adding New Features

1. **Backend API**: Add endpoints in `backend/app/main.py` or create new routers
2. **Database**: Modify `infra/init.sql` for schema changes
3. **RAG**: Enhance `backend/app/rag.py` for improved search
4. **UI**: Update `ui_app/streamlit_app.py` for new interface features

### Running Tests
```bash
# Inside backend container
docker exec -it backend pytest
```

## Performance Optimization

### Database Tuning
```sql
-- Optimize vector search index
CREATE INDEX IF NOT EXISTS idx_chunks_ivfflat
ON chunks USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);

-- Analyze tables for query planning
ANALYZE chunks;
ANALYZE datasets;
```

### Increase Rate Limits
1. Register for a Socrata app token: https://dev.socrata.com/docs/app-tokens.html
2. Add to `.env`: `SOCRATA_APP_TOKEN=your-token`

### Batch Processing
For large-scale ingestion, use the batch indexer:
```bash
python index_cdc_mental_health_intelligent.py
```

## Security Considerations

1. **Change default passwords** in `.env` before production deployment
2. **Secure your OpenAI API key** - never commit it to version control
3. **Use environment variables** for all sensitive configuration
4. **Enable authentication** if exposing the platform publicly
5. **Regular backups** of PostgreSQL data volume
6. **Network isolation** - consider using Docker networks in production

## License

See LICENSE file for details.

## Support

For issues and questions:
1. Check logs: `docker-compose logs [service]`
2. Review this README's troubleshooting section
3. Check API documentation: http://localhost:8000/docs
4. Verify environment configuration in `.env`

## Acknowledgments

- Data sources: CDC and SAMHSA Socrata Open Data platforms
- Vector search: pgvector extension for PostgreSQL
- LLM: OpenAI API for RAG and question answering
- Storage: MinIO for S3-compatible object storage
