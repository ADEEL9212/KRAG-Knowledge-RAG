# KRAG Quick Start Guide

This guide will help you get started with the KRAG (Knowledge Retrieval-Augmented Generation) system quickly.

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- OpenAI API key (for answer synthesis) - Optional for basic functionality

## Installation

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/ADEEL9212/KRAG-Knowledge-RAG.git
cd KRAG-Knowledge-RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the template
cp .env.template .env

# Edit .env and add your OpenAI API key (if you want LLM-powered answers)
# OPENAI_API_KEY=your_key_here
```

## Testing the Installation

Run the example test script to verify everything is working:

```bash
python examples/test_system.py
```

You should see output like:
```
================================================================================
KRAG SYSTEM TEST
================================================================================
✓ Configuration PASSED
✓ Document Chunker PASSED
TEST RESULTS: 2 passed, 0 failed
```

## Running the API Server

### Start the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### Access the Interactive Documentation

Open your browser and navigate to:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Basic Usage

### 1. Upload Documents

Upload documents to build your knowledge base:

```bash
curl -X POST \
  -F "files=@examples/sample_document.txt" \
  http://localhost:8000/api/documents/upload
```

Expected response:
```json
{
  "status": "success",
  "files_processed": 1,
  "chunks_created": 5,
  "message": "Successfully processed 1/1 files",
  "errors": []
}
```

### 2. Query Without LLM (Retrieval Only)

Get relevant document chunks without AI synthesis:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "top_k": 3,
    "use_llm": false
  }' \
  http://localhost:8000/api/query
```

### 3. Query With LLM (Full RAG)

Get AI-generated answers based on retrieved documents:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the benefits of RAG?",
    "top_k": 5,
    "use_llm": true
  }' \
  http://localhost:8000/api/query
```

### 4. Health Check

```bash
curl http://localhost:8000/health
```

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start
cd docker
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Using Docker Directly

```bash
# Build image
docker build -f docker/Dockerfile -t krag:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key_here \
  -v $(pwd)/data:/app/data \
  krag:latest
```

## Common Workflows

### Workflow 1: Building a Knowledge Base

1. Collect your documents (PDF, DOCX, TXT)
2. Start the API server
3. Upload documents via `/api/documents/upload`
4. Query the knowledge base via `/api/query`

### Workflow 2: Testing Different Chunking Strategies

Edit `.env` to change chunking parameters:
```bash
CHUNK_SIZE=1000          # Increase for larger context
CHUNK_OVERLAP=100        # Increase for better continuity
```

### Workflow 3: Using Different Embedding Models

Change the embedding model in `.env`:
```bash
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

Note: Different models have different dimensions. Make sure to clear the vector store when changing models.

## Troubleshooting

### Issue: Module not found errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: ChromaDB persistence errors

Check that the data directory exists and is writable:
```bash
mkdir -p ./data/chroma
chmod 755 ./data/chroma
```

### Issue: OpenAI API errors

- Verify your API key is correct in `.env`
- Check your OpenAI account has credits
- Set `use_llm: false` in queries to bypass LLM (retrieval only)

### Issue: Out of memory

- Reduce `CHUNK_SIZE` in `.env`
- Reduce `top_k` in your queries
- Process fewer documents at once

## Advanced Configuration

### Using Pinecone Instead of ChromaDB

1. Set up Pinecone account and get API key
2. Update `.env`:
```bash
VECTOR_STORE_TYPE=pinecone
PINECONE_API_KEY=your_key
PINECONE_ENVIRONMENT=your_env
PINECONE_INDEX_NAME=krag-index
```

Note: Full Pinecone implementation is a stub. Implement the methods in `app/vector_store/pinecone.py`.

### Custom Ranking Strategies

Modify the ranker initialization in `app/main.py`:
```python
ranker = DocumentRanker(strategy="mmr")  # Options: similarity, diversity, mmr
```

### Logging Configuration

Change log level and format in `.env`:
```bash
LOG_LEVEL=DEBUG          # Options: DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=json          # Options: text, json
```

## Performance Tips

1. **Batch Upload**: Upload multiple files at once for better efficiency
2. **Optimize Chunk Size**: Balance between context (larger) and precision (smaller)
3. **Use Similarity Threshold**: Filter low-relevance results with `similarity_threshold`
4. **Persistent Storage**: Use ChromaDB persistence for production to avoid re-indexing
5. **Docker Volumes**: Mount data directory to persist between container restarts

## Next Steps

- Read the [README.md](README.md) for detailed architecture information
- Explore the [API documentation](http://localhost:8000/docs)
- Check [examples/](examples/) for sample documents and scripts
- Review the code in [app/](app/) to understand the implementation
- Customize the system for your specific use case

## Getting Help

- Check the [README.md](README.md) for comprehensive documentation
- Review the code comments and docstrings
- Open an issue on GitHub for bugs or feature requests

## License

MIT License - see [LICENSE](LICENSE) for details
