"""
Tests for vector store components.
"""

import pytest
import numpy as np
from app.vector_store import ChromaVectorStore


class TestChromaVectorStore:
    """Tests for ChromaVectorStore."""

    @pytest.fixture
    def vector_store(self):
        """Create a test vector store (in-memory)."""
        return ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=None,  # In-memory for testing
        )

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings."""
        return np.random.rand(3, 384).astype(np.float32)

    def test_add_documents(self, vector_store, sample_embeddings):
        """Test adding documents to vector store."""
        texts = ["First document", "Second document", "Third document"]
        metadatas = [
            {"source": "test1"},
            {"source": "test2"},
            {"source": "test3"},
        ]

        ids = vector_store.add_documents(
            texts=texts,
            embeddings=sample_embeddings,
            metadatas=metadatas,
        )

        assert len(ids) == 3
        assert all(isinstance(id, str) for id in ids)

    def test_search(self, vector_store, sample_embeddings):
        """Test searching the vector store."""
        texts = ["Document about Python", "Document about Java", "Document about C++"]
        metadatas = [{"lang": "python"}, {"lang": "java"}, {"lang": "cpp"}]

        vector_store.add_documents(
            texts=texts,
            embeddings=sample_embeddings,
            metadatas=metadatas,
        )

        # Search with first embedding
        query_embedding = sample_embeddings[0]
        results = vector_store.search(query_embedding, top_k=2)

        assert len(results) <= 2
        assert all("content" in r for r in results)
        assert all("score" in r for r in results)
        assert all("metadata" in r for r in results)

    def test_delete(self, vector_store, sample_embeddings):
        """Test deleting documents."""
        texts = ["Doc 1", "Doc 2", "Doc 3"]

        ids = vector_store.add_documents(
            texts=texts,
            embeddings=sample_embeddings,
        )

        # Delete first document
        success = vector_store.delete([ids[0]])
        assert success is True

    def test_get_collection_stats(self, vector_store, sample_embeddings):
        """Test getting collection statistics."""
        texts = ["Doc 1", "Doc 2"]
        embeddings = sample_embeddings[:2]

        vector_store.add_documents(texts=texts, embeddings=embeddings)

        stats = vector_store.get_collection_stats()

        assert "document_count" in stats
        assert stats["document_count"] >= 2

    def test_clear_collection(self, vector_store, sample_embeddings):
        """Test clearing the collection."""
        texts = ["Doc 1", "Doc 2"]
        embeddings = sample_embeddings[:2]

        vector_store.add_documents(texts=texts, embeddings=embeddings)
        success = vector_store.clear_collection()

        assert success is True

        stats = vector_store.get_collection_stats()
        assert stats["document_count"] == 0
