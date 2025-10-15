"""
Document chunker for splitting text into manageable chunks.
"""

import re
from typing import List, Dict, Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """Chunker for splitting documents into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: str = "character",
    ):
        """
        Initialize the document chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            strategy: Chunking strategy ('character', 'sentence', 'paragraph')
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        logger.info(
            f"Initialized DocumentChunker with size={chunk_size}, "
            f"overlap={chunk_overlap}, strategy={strategy}"
        )

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Split text into chunks with metadata.

        Args:
            text: Text content to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunks with metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        if self.strategy == "character":
            chunks = self._chunk_by_character(text)
        elif self.strategy == "sentence":
            chunks = self._chunk_by_sentence(text)
        elif self.strategy == "paragraph":
            chunks = self._chunk_by_paragraph(text)
        else:
            logger.warning(f"Unknown strategy {self.strategy}, using character")
            chunks = self._chunk_by_character(text)

        # Add metadata to each chunk
        result = []
        for idx, chunk_text in enumerate(chunks):
            chunk_data = {
                "content": chunk_text,
                "chunk_index": idx,
                "metadata": metadata.copy() if metadata else {},
            }
            result.append(chunk_data)

        logger.info(f"Created {len(result)} chunks from text")
        return result

    def _chunk_by_character(self, text: str) -> List[str]:
        """
        Split text by character count with overlap.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size

            # Try to break at sentence boundary if possible
            if end < text_length:
                # Look for sentence endings within the chunk
                chunk_text = text[start:end]
                # Find last sentence boundary
                last_period = chunk_text.rfind(". ")
                last_newline = chunk_text.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > self.chunk_size // 2:  # Only if reasonable
                    end = start + break_point + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position accounting for overlap
            start = end - self.chunk_overlap

        return chunks

    def _chunk_by_sentence(self, text: str) -> List[str]:
        """
        Split text by sentences, respecting chunk size.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                # Keep last few sentences for overlap
                overlap_text = " ".join(current_chunk)
                if len(overlap_text) > self.chunk_overlap:
                    # Find sentences that fit in overlap
                    overlap_sentences = []
                    overlap_size = 0
                    for s in reversed(current_chunk):
                        if overlap_size + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_size += len(s)
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size

        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _chunk_by_paragraph(self, text: str) -> List[str]:
        """
        Split text by paragraphs, respecting chunk size.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = []
        current_size = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            para_size = len(paragraph)

            # If single paragraph exceeds chunk size, split it
            if para_size > self.chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split large paragraph using character chunking
                para_chunks = self._chunk_by_character(paragraph)
                chunks.extend(para_chunks)
                continue

            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))

                # Overlap handling
                if current_size > self.chunk_overlap:
                    current_chunk = [current_chunk[-1]]
                    current_size = len(current_chunk[-1])
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(paragraph)
            current_size += para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks
