#!/usr/bin/env python
"""
Simple test script to demonstrate the KRAG system without requiring external dependencies.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.document_processor.chunker import DocumentChunker
from app.utils.logger import setup_logger

# Setup logging
logger = setup_logger("test", level="INFO", log_format="text")

def test_chunker():
    """Test the document chunker."""
    logger.info("Testing DocumentChunker...")
    
    # Read sample document
    sample_file = os.path.join(os.path.dirname(__file__), "sample_document.txt")
    
    if os.path.exists(sample_file):
        with open(sample_file, 'r') as f:
            content = f.read()
    else:
        content = "This is a test document. " * 50
    
    # Create chunker
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    
    # Chunk the document
    chunks = chunker.chunk(content, metadata={"source": "sample_document.txt"})
    
    logger.info(f"✓ Created {len(chunks)} chunks from document")
    logger.info(f"✓ First chunk length: {len(chunks[0]['content'])} characters")
    logger.info(f"✓ Metadata preserved: {chunks[0]['metadata']}")
    
    # Display first chunk
    print("\n" + "="*80)
    print("FIRST CHUNK PREVIEW:")
    print("="*80)
    print(chunks[0]['content'][:200] + "...")
    print("="*80 + "\n")
    
    return True

def test_config():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    
    from app.config import settings
    
    logger.info(f"✓ Chunk size: {settings.chunk_size}")
    logger.info(f"✓ Embedding model: {settings.embedding_model}")
    logger.info(f"✓ Vector store type: {settings.vector_store_type}")
    logger.info(f"✓ Top K: {settings.top_k}")
    
    return True

def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("KRAG SYSTEM TEST")
    print("="*80 + "\n")
    
    tests = [
        ("Configuration", test_config),
        ("Document Chunker", test_chunker),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\nRunning test: {test_name}")
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED\n")
            else:
                failed += 1
                logger.error(f"✗ {test_name} FAILED\n")
        except Exception as e:
            failed += 1
            logger.error(f"✗ {test_name} FAILED: {str(e)}\n")
    
    print("\n" + "="*80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*80 + "\n")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
