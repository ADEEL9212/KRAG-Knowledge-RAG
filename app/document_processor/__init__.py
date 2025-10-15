"""
Document processing pipeline for parsing, chunking, and embedding documents.
"""

# Lazy imports to avoid import errors when dependencies are missing
__all__ = ["DocumentParser", "DocumentChunker", "DocumentEmbedder"]


def __getattr__(name):
    """Lazy import to avoid loading all dependencies at once."""
    if name == "DocumentParser":
        from .parser import DocumentParser
        return DocumentParser
    elif name == "DocumentChunker":
        from .chunker import DocumentChunker
        return DocumentChunker
    elif name == "DocumentEmbedder":
        from .embedder import DocumentEmbedder
        return DocumentEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
