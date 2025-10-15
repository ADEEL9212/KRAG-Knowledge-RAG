"""
FastAPI route definitions for the KRAG API.
"""

import os
import time
import tempfile
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pathlib import Path

from app.api.models import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    HealthResponse,
    SourceDocument,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Global state (will be injected via dependencies)
_app_state = {}


def get_app_state():
    """Dependency to get application state."""
    return _app_state


def set_app_state(state: dict):
    """Set application state (called during startup)."""
    global _app_state
    _app_state = state


@router.post("/api/documents/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    state: dict = Depends(get_app_state),
):
    """
    Upload and process documents to add to the knowledge base.

    Args:
        files: List of uploaded files
        state: Application state with processor and vector store

    Returns:
        Upload status and statistics
    """
    logger.info(f"Received {len(files)} files for upload")

    parser = state.get("parser")
    chunker = state.get("chunker")
    embedder = state.get("embedder")
    vector_store = state.get("vector_store")

    if not all([parser, chunker, embedder, vector_store]):
        raise HTTPException(
            status_code=500, detail="Server not properly initialized"
        )

    processed_files = 0
    total_chunks = 0
    errors = []

    # Create temporary directory for uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            try:
                # Save uploaded file temporarily
                file_path = Path(temp_dir) / file.filename
                content = await file.read()

                with open(file_path, "wb") as f:
                    f.write(content)

                logger.info(f"Processing file: {file.filename}")

                # Parse document
                parsed_doc = parser.parse(str(file_path))

                # Chunk document
                chunks = chunker.chunk(
                    parsed_doc["content"], parsed_doc["metadata"]
                )

                if not chunks:
                    logger.warning(f"No chunks created for {file.filename}")
                    errors.append(f"{file.filename}: No content extracted")
                    continue

                # Generate embeddings
                chunk_texts = [chunk["content"] for chunk in chunks]
                embeddings = embedder.embed_batch(chunk_texts)

                # Prepare metadata
                metadatas = [chunk["metadata"] for chunk in chunks]

                # Add to vector store
                vector_store.add_documents(
                    texts=chunk_texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )

                processed_files += 1
                total_chunks += len(chunks)

                logger.info(
                    f"Successfully processed {file.filename}: {len(chunks)} chunks"
                )

            except Exception as e:
                error_msg = f"{file.filename}: {str(e)}"
                logger.error(f"Error processing file: {error_msg}")
                errors.append(error_msg)

    status = "success" if processed_files > 0 else "failed"
    message = f"Successfully processed {processed_files}/{len(files)} files"

    return UploadResponse(
        status=status,
        files_processed=processed_files,
        chunks_created=total_chunks,
        message=message,
        errors=errors,
    )


@router.post("/api/query", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    state: dict = Depends(get_app_state),
):
    """
    Query the knowledge base and get AI-generated answers.

    Args:
        request: Query request with parameters
        state: Application state with retriever and synthesizer

    Returns:
        Query results with answer and sources
    """
    logger.info(f"Processing query: {request.query[:100]}...")

    retriever = state.get("retriever")
    ranker = state.get("ranker")
    synthesizer = state.get("synthesizer")

    if not all([retriever, ranker]):
        raise HTTPException(
            status_code=500, detail="Server not properly initialized"
        )

    try:
        # Retrieve documents
        start_time = time.time()
        documents = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata,
        )
        retrieval_time = time.time() - start_time

        # Filter by threshold
        if request.similarity_threshold > 0:
            documents = ranker.filter_by_threshold(
                documents, request.similarity_threshold
            )

        # Rank documents
        documents = ranker.rank(documents, query=request.query)

        # Format sources
        sources = [
            SourceDocument(
                content=doc["content"],
                metadata=doc["metadata"],
                score=doc["score"],
            )
            for doc in documents
        ]

        # Generate answer if requested
        answer = None
        synthesis_time = None
        model = None

        if request.use_llm and synthesizer:
            start_time = time.time()
            synthesis_result = synthesizer.synthesize(
                query=request.query,
                documents=documents,
                include_sources=False,
            )
            synthesis_time = time.time() - start_time

            answer = synthesis_result.get("answer")
            model = synthesis_result.get("model")

        logger.info(
            f"Query completed: {len(sources)} sources, "
            f"retrieval={retrieval_time:.2f}s"
        )

        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            retrieval_time=retrieval_time,
            synthesis_time=synthesis_time,
            model=model,
        )

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check(state: dict = Depends(get_app_state)):
    """
    Health check endpoint.

    Args:
        state: Application state

    Returns:
        Health status and version information
    """
    from app import __version__

    vector_store = state.get("vector_store")
    vector_store_stats = {}

    if vector_store:
        try:
            vector_store_stats = vector_store.get_collection_stats()
        except Exception as e:
            logger.warning(f"Could not get vector store stats: {str(e)}")

    return HealthResponse(
        status="healthy",
        version=__version__,
        vector_store=vector_store_stats,
    )
