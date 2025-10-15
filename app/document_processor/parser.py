"""
Document parser for extracting text from multiple file formats.
"""

import os
from pathlib import Path
from typing import Dict, Optional, List
import fitz  # PyMuPDF
import docx2txt
from pypdf import PdfReader

from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentParser:
    """Parser for extracting text from various document formats."""

    SUPPORTED_FORMATS = {".pdf", ".docx", ".txt"}

    def __init__(self):
        """Initialize the document parser."""
        logger.info("Initialized DocumentParser")

    def parse(self, file_path: str) -> Dict[str, any]:
        """
        Parse a document and extract text with metadata.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary containing text content and metadata

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = file_path_obj.suffix.lower()

        if file_extension not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        logger.info(f"Parsing document: {file_path}")

        try:
            if file_extension == ".pdf":
                content = self._parse_pdf(file_path)
            elif file_extension == ".docx":
                content = self._parse_docx(file_path)
            elif file_extension == ".txt":
                content = self._parse_txt(file_path)
            else:
                raise ValueError(f"Unsupported format: {file_extension}")

            metadata = {
                "filename": file_path_obj.name,
                "file_path": str(file_path_obj.absolute()),
                "file_type": file_extension[1:],
                "file_size": file_path_obj.stat().st_size,
            }

            logger.info(
                f"Successfully parsed {file_path_obj.name}: "
                f"{len(content)} characters"
            )

            return {"content": content, "metadata": metadata}

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            raise

    def _parse_pdf(self, file_path: str) -> str:
        """
        Parse PDF file using PyMuPDF for better quality extraction.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text content
        """
        text_content = []

        try:
            # Use PyMuPDF for better extraction quality
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text()
                    if text.strip():
                        text_content.append(text)

            return "\n\n".join(text_content)

        except Exception as e:
            logger.warning(f"PyMuPDF failed, falling back to pypdf: {str(e)}")
            # Fallback to pypdf
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
                return "\n\n".join(text_content)
            except Exception as e2:
                logger.error(f"Both PDF parsers failed: {str(e2)}")
                raise

    def _parse_docx(self, file_path: str) -> str:
        """
        Parse DOCX file.

        Args:
            file_path: Path to DOCX file

        Returns:
            Extracted text content
        """
        try:
            text = docx2txt.process(file_path)
            return text
        except Exception as e:
            logger.error(f"Error parsing DOCX: {str(e)}")
            raise

    def _parse_txt(self, file_path: str) -> str:
        """
        Parse plain text file.

        Args:
            file_path: Path to TXT file

        Returns:
            File content
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error parsing TXT: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Error parsing TXT: {str(e)}")
            raise

    def parse_batch(self, file_paths: List[str]) -> List[Dict[str, any]]:
        """
        Parse multiple documents.

        Args:
            file_paths: List of file paths to parse

        Returns:
            List of parsed documents with metadata
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.parse(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {str(e)}")
                # Continue with other files
                continue

        logger.info(f"Successfully parsed {len(results)}/{len(file_paths)} documents")
        return results
