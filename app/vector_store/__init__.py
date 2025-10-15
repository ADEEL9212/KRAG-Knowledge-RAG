"""
Vector store implementations for storing and retrieving document embeddings.
"""

from .base import VectorStore
from .chroma import ChromaVectorStore

__all__ = ["VectorStore", "ChromaVectorStore"]
