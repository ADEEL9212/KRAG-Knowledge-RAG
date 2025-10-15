"""
Tests for document processing components.
"""

import pytest
from app.document_processor import DocumentChunker, DocumentEmbedder


class TestDocumentChunker:
    """Tests for DocumentChunker."""

    def test_chunk_by_character(self):
        """Test character-based chunking."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a test. " * 10
        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all("chunk_index" in chunk for chunk in chunks)

    def test_chunk_with_metadata(self):
        """Test chunking with metadata preservation."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "Test text content."
        metadata = {"filename": "test.txt", "source": "test"}

        chunks = chunker.chunk(text, metadata)

        assert len(chunks) > 0
        assert chunks[0]["metadata"]["filename"] == "test.txt"
        assert chunks[0]["metadata"]["source"] == "test"

    def test_empty_text(self):
        """Test handling of empty text."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk("")

        assert len(chunks) == 0

    def test_chunk_overlap_validation(self):
        """Test that overlap must be less than chunk size."""
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=50, chunk_overlap=60)


class TestDocumentEmbedder:
    """Tests for DocumentEmbedder."""

    def test_embed_single_text(self):
        """Test embedding a single text."""
        embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2")
        text = "This is a test sentence."

        embedding = embedder.embed(text)

        assert embedding is not None
        assert len(embedding.shape) == 2
        assert embedding.shape[0] == 1
        assert embedding.shape[1] == embedder.get_embedding_dimension()

    def test_embed_multiple_texts(self):
        """Test embedding multiple texts."""
        embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2")
        texts = ["First sentence.", "Second sentence.", "Third sentence."]

        embeddings = embedder.embed(texts)

        assert embeddings is not None
        assert len(embeddings.shape) == 2
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == embedder.get_embedding_dimension()

    def test_embed_batch(self):
        """Test batch embedding."""
        embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2")
        texts = ["Text one.", "Text two.", "Text three."]

        embeddings = embedder.embed_batch(texts, batch_size=2)

        assert embeddings is not None
        assert embeddings.shape[0] == 3

    def test_embedding_dimension(self):
        """Test getting embedding dimension."""
        embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2")
        dimension = embedder.get_embedding_dimension()

        assert dimension == 384  # Known dimension for all-MiniLM-L6-v2
