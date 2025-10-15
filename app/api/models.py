"""
Pydantic models for API requests and responses.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for querying the knowledge base."""

    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(default=5, description="Number of results to retrieve", ge=1, le=50)
    use_llm: bool = Field(default=True, description="Whether to use LLM for answer synthesis")
    similarity_threshold: float = Field(
        default=0.0, description="Minimum similarity threshold", ge=0.0, le=1.0
    )
    filter_metadata: Optional[Dict] = Field(
        default=None, description="Metadata filters for retrieval"
    )


class SourceDocument(BaseModel):
    """Model for source document in response."""

    content: str = Field(..., description="Document content")
    metadata: Dict = Field(default_factory=dict, description="Document metadata")
    score: float = Field(..., description="Similarity score")


class QueryResponse(BaseModel):
    """Response model for query results."""

    query: str = Field(..., description="Original query")
    answer: Optional[str] = Field(default=None, description="Generated answer")
    sources: List[SourceDocument] = Field(
        default_factory=list, description="Source documents"
    )
    retrieval_time: float = Field(..., description="Time taken for retrieval (seconds)")
    synthesis_time: Optional[float] = Field(
        default=None, description="Time taken for synthesis (seconds)"
    )
    model: Optional[str] = Field(default=None, description="LLM model used")


class UploadResponse(BaseModel):
    """Response model for document upload."""

    status: str = Field(..., description="Upload status")
    files_processed: int = Field(..., description="Number of files processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    message: str = Field(..., description="Status message")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Application version")
    vector_store: Dict = Field(default_factory=dict, description="Vector store statistics")
