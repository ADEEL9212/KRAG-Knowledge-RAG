# KRAG - Knowledge Retrieval-Augmented Generation

A production-ready, Python-based knowledge-base search engine using Retrieval-Augmented Generation (RAG) technology. This system enables semantic search over multiple document formats with AI-powered answer synthesis.

## 🌟 Features

- **Multi-Format Support**: Process PDF, DOCX, and TXT documents
- **Smart Chunking**: Configurable text chunking with overlap for optimal context preservation
- **Vector Search**: ChromaDB integration for efficient semantic search
- **AI-Powered Answers**: OpenAI GPT integration for answer synthesis
- **RESTful API**: FastAPI-based REST endpoints for easy integration
- **Docker Ready**: Containerized deployment for production environments
- **Extensible**: Modular architecture supporting multiple vector stores and LLM providers

## 🏗️ Architecture

```
knowledge_rag/
├── app/
│   ├── document_processor/  # Document ingestion pipeline
│   │   ├── parser.py        # Multi-format document parsing
│   │   ├── chunker.py       # Text chunking strategies
│   │   └── embedder.py      # Embedding generation
│   ├── vector_store/        # Vector database layer
│   │   ├── base.py          # Abstract interface
│   │   ├── chroma.py        # ChromaDB implementation
│   │   └── pinecone.py      # Pinecone implementation
│   ├── query_engine/        # Retrieval and synthesis
│   │   ├── retriever.py     # Document retrieval
│   │   ├── ranker.py        # Result ranking
│   │   └── synthesizer.py   # Answer generation
│   ├── api/                 # API layer
│   │   ├── routes.py        # FastAPI routes
│   │   └── models.py        # Pydantic models
│   ├── utils/               # Utilities
│   │   └── logger.py        # Logging setup
│   ├── config.py            # Configuration management
│   └── main.py              # Application entry point
├── tests/                   # Test suite
├── docker/                  # Docker configuration
└── requirements.txt         # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key (for answer synthesis)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/ADEEL9212/KRAG-Knowledge-RAG.git
cd KRAG-Knowledge-RAG
```

2. **Create and activate a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment**:
```bash
cp .env.template .env
# Edit .env and add your API keys
```

5. **Start the server**:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

## 📚 Usage

### Upload Documents

Upload documents to build your knowledge base:

```bash
curl -X POST \
  -F "files=@document.pdf" \
  http://localhost:8000/api/documents/upload
```

### Query the Knowledge Base

Ask questions and get AI-powered answers:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "top_k": 5,
    "use_llm": true
  }' \
  http://localhost:8000/api/query
```

### Health Check

```bash
curl http://localhost:8000/health
```

## 🐳 Docker Deployment

### Build and run with Docker Compose:

```bash
cd docker
docker-compose up -d
```

### Build Docker image:

```bash
docker build -f docker/Dockerfile -t knowledge-rag:latest .
```

### Run container:

```bash
docker run -p 8000:8000 --env-file .env knowledge-rag:latest
```

## 🔧 Configuration

Configuration is managed through environment variables. Key settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `VECTOR_STORE_TYPE` | Vector store backend | `chroma` |
| `EMBEDDING_MODEL` | Sentence Transformer model | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Text chunk size | `500` |
| `CHUNK_OVERLAP` | Chunk overlap | `50` |
| `TOP_K` | Number of results to retrieve | `5` |
| `LLM_MODEL` | OpenAI model | `gpt-3.5-turbo` |

See `.env.template` for all available options.

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_document_processor.py
```

## 📖 API Documentation

### Endpoints

#### POST `/api/documents/upload`
Upload one or multiple documents to the knowledge base.

**Request**: Multipart form data with `files` field

**Response**:
```json
{
  "status": "success",
  "files_processed": 1,
  "chunks_created": 42,
  "message": "Documents processed successfully"
}
```

#### POST `/api/query`
Query the knowledge base with semantic search.

**Request**:
```json
{
  "query": "Your question here",
  "top_k": 5,
  "use_llm": true,
  "similarity_threshold": 0.7
}
```

**Response**:
```json
{
  "query": "Your question here",
  "answer": "AI-generated answer based on context",
  "sources": [
    {
      "content": "Relevant text chunk",
      "metadata": {
        "filename": "document.pdf",
        "page": 1
      },
      "score": 0.92
    }
  ],
  "retrieval_time": 0.15,
  "synthesis_time": 1.23
}
```

#### GET `/health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## 🔍 How It Works

1. **Document Ingestion**:
   - Documents are parsed into text using format-specific parsers
   - Text is split into overlapping chunks for better context
   - Chunks are embedded using Sentence Transformers
   - Embeddings are stored in ChromaDB with metadata

2. **Query Processing**:
   - User query is embedded using the same model
   - Vector similarity search retrieves relevant chunks
   - Results are ranked by relevance score
   - LLM synthesizes a coherent answer from top chunks

3. **Answer Generation**:
   - Retrieved chunks provide context to the LLM
   - Custom prompt engineering ensures accurate responses
   - Sources are cited for transparency

## 🛠️ Technology Stack

- **API Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Document Processing**: [PyMuPDF](https://pymupdf.readthedocs.io/), [python-docx](https://python-docx.readthedocs.io/)
- **Embeddings**: [Sentence Transformers](https://www.sbert.net/)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **LLM**: [OpenAI GPT](https://platform.openai.com/docs/)
- **Containerization**: [Docker](https://www.docker.com/)

## 🚧 Future Enhancements

- [ ] Hybrid search (dense + sparse retrievers)
- [ ] Multi-query retrieval with query reformulation
- [ ] Semantic chunking by topic
- [ ] Caching for frequent queries
- [ ] Authentication and authorization
- [ ] Rate limiting
- [ ] Streaming responses
- [ ] Multi-turn conversation support
- [ ] Fine-tuning pipeline for domain-specific embeddings
- [ ] Document versioning and refresh
- [ ] Multi-modal support (images, tables)

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🙏 Acknowledgments

Built with modern RAG techniques and best practices from the AI/ML community.