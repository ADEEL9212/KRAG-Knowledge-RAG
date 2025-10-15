"""
Main FastAPI application entry point.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.utils.logger import setup_logger, get_logger
from app.document_processor import DocumentParser, DocumentChunker, DocumentEmbedder
from app.vector_store import ChromaVectorStore
from app.query_engine import DocumentRetriever, DocumentRanker, AnswerSynthesizer
from app.api.routes import router, set_app_state

# Setup logging
setup_logger(
    name="krag",
    level=settings.log_level,
    log_format=settings.log_format,
)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting KRAG application...")

    # Initialize components
    try:
        # Document processing components
        parser = DocumentParser()
        chunker = DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        embedder = DocumentEmbedder(model_name=settings.embedding_model)

        # Vector store
        if settings.vector_store_type.lower() == "chroma":
            # Create persist directory if it doesn't exist
            os.makedirs(settings.chroma_persist_directory, exist_ok=True)
            vector_store = ChromaVectorStore(
                collection_name="knowledge_base",
                persist_directory=settings.chroma_persist_directory,
            )
        else:
            logger.error(f"Unsupported vector store: {settings.vector_store_type}")
            raise ValueError(
                f"Unsupported vector store type: {settings.vector_store_type}"
            )

        # Query engine components
        retriever = DocumentRetriever(
            vector_store=vector_store,
            embedder=embedder,
            top_k=settings.top_k,
        )
        ranker = DocumentRanker(strategy="similarity")

        # Answer synthesizer (optional, requires API key)
        synthesizer = None
        if settings.openai_api_key:
            synthesizer = AnswerSynthesizer(
                api_key=settings.openai_api_key,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.max_tokens,
            )
        else:
            logger.warning(
                "No OpenAI API key provided. Answer synthesis will be disabled."
            )

        # Set application state
        app_state = {
            "parser": parser,
            "chunker": chunker,
            "embedder": embedder,
            "vector_store": vector_store,
            "retriever": retriever,
            "ranker": ranker,
            "synthesizer": synthesizer,
        }
        set_app_state(app_state)

        logger.info("KRAG application started successfully")

    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down KRAG application...")


# Create FastAPI app
app = FastAPI(
    title="KRAG - Knowledge Retrieval-Augmented Generation",
    description="A production-ready RAG system for semantic search and AI-powered answers",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "KRAG - Knowledge Retrieval-Augmented Generation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
