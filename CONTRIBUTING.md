# Contributing to KRAG

Thank you for your interest in contributing to KRAG (Knowledge Retrieval-Augmented Generation)! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the problem
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, etc.)
- Any relevant logs or error messages

### Suggesting Features

We welcome feature suggestions! Please open an issue with:
- A clear description of the feature
- Use cases and benefits
- Any implementation ideas you have

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Test your changes**: Ensure all tests pass
5. **Commit your changes**: Use clear, descriptive commit messages
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Open a pull request**

## Development Setup

### Setting Up Your Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/KRAG-Knowledge-RAG.git
cd KRAG-Knowledge-RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install black flake8 mypy pytest pytest-cov
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_document_processor.py

# Run specific test
pytest tests/test_document_processor.py::TestDocumentChunker::test_chunk_by_character
```

### Code Style

We follow PEP 8 style guidelines. Use the following tools:

```bash
# Format code with black
black app/ tests/

# Check style with flake8
flake8 app/ tests/

# Type checking with mypy
mypy app/
```

### Project Structure

```
app/
├── document_processor/  # Document ingestion
├── vector_store/        # Vector database
├── query_engine/        # Retrieval and synthesis
├── api/                 # REST API
├── utils/               # Utilities
├── config.py            # Configuration
└── main.py              # Application entry
```

## Coding Guidelines

### Python Style

- Follow PEP 8
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and under 50 lines when possible
- Use meaningful variable and function names

### Example Code Style

```python
from typing import List, Dict, Optional

def process_documents(
    documents: List[str],
    chunk_size: int = 500,
    metadata: Optional[Dict] = None,
) -> List[Dict]:
    """
    Process documents into chunks.

    Args:
        documents: List of document contents
        chunk_size: Maximum chunk size in characters
        metadata: Optional metadata to attach

    Returns:
        List of processed chunks with metadata
    """
    # Implementation here
    pass
```

### Commit Messages

Use clear, descriptive commit messages:

```
Good:
- Add support for EPUB documents
- Fix chunking overlap calculation
- Improve error handling in parser

Bad:
- Update
- Fix bug
- Changes
```

### Documentation

- Update README.md if adding new features
- Add docstrings to new functions and classes
- Update examples if changing APIs
- Keep QUICKSTART.md current

## Areas for Contribution

### High Priority

1. **Additional Document Formats**: Add support for more file types (HTML, EPUB, Markdown)
2. **Pinecone Integration**: Complete the Pinecone vector store implementation
3. **Streaming Responses**: Implement streaming for LLM responses
4. **Evaluation Metrics**: Add retrieval and answer quality evaluation
5. **Authentication**: Add API authentication and rate limiting

### Medium Priority

1. **Hybrid Search**: Implement dense + sparse retrieval
2. **Multi-query Retrieval**: Query reformulation and expansion
3. **Semantic Chunking**: Chunk by topic/semantic boundaries
4. **Conversation History**: Support multi-turn conversations
5. **Fine-tuning Pipeline**: Add embedding fine-tuning capability

### Nice to Have

1. **Multi-modal Support**: Handle images and tables
2. **User Feedback Loop**: Capture and use user feedback
3. **Document Versioning**: Track document updates
4. **Custom Embeddings**: Support for custom embedding models
5. **Advanced Ranking**: Implement cross-encoder re-ranking

## Adding New Features

### Adding a New Document Format

1. Update `app/document_processor/parser.py`
2. Add parsing logic for the new format
3. Update `SUPPORTED_FORMATS` set
4. Add tests in `tests/test_document_processor.py`
5. Update documentation

### Adding a New Vector Store

1. Create new file in `app/vector_store/`
2. Inherit from `VectorStore` base class
3. Implement all abstract methods
4. Add configuration in `app/config.py`
5. Update `app/main.py` to support new store
6. Add tests

### Adding a New API Endpoint

1. Add Pydantic models in `app/api/models.py`
2. Add route in `app/api/routes.py`
3. Update OpenAPI documentation
4. Add tests in `tests/test_api.py`
5. Update API documentation

## Testing Guidelines

### Writing Tests

- Test public interfaces, not implementation details
- Use fixtures for common setup
- Mock external dependencies (OpenAI, etc.)
- Aim for >80% code coverage
- Test edge cases and error conditions

### Test Structure

```python
import pytest
from app.document_processor import DocumentChunker

class TestDocumentChunker:
    """Tests for DocumentChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a test chunker."""
        return DocumentChunker(chunk_size=100, chunk_overlap=20)

    def test_chunk_by_character(self, chunker):
        """Test character-based chunking."""
        text = "Test text " * 20
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
```

## Review Process

1. All pull requests require review
2. CI checks must pass (when implemented)
3. Code must follow style guidelines
4. Tests must be included for new features
5. Documentation must be updated

## Questions?

- Open an issue for questions
- Check existing issues and discussions
- Review documentation in README.md and QUICKSTART.md

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

Thank you for contributing to KRAG! 🚀
