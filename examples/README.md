# KRAG Examples

This directory contains example files and scripts to demonstrate the KRAG system.

## Files

- **sample_document.txt**: A sample document about RAG (Retrieval-Augmented Generation) that can be used for testing.
- **test_system.py**: A simple test script that demonstrates the document chunking functionality.

## Running the Examples

### Test the System

Run the test script to verify basic functionality:

```bash
python examples/test_system.py
```

This will:
- Load configuration settings
- Test the document chunker with the sample document
- Display the first chunk of the processed document

### Using with the API

Once you have the API running, you can upload the sample document:

```bash
# Start the API server
uvicorn app.main:app --reload

# Upload the sample document (in another terminal)
curl -X POST \
  -F "files=@examples/sample_document.txt" \
  http://localhost:8000/api/documents/upload

# Query the knowledge base
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "top_k": 3, "use_llm": false}' \
  http://localhost:8000/api/query
```

## Adding Your Own Documents

You can add your own PDF, DOCX, or TXT documents to this directory and upload them using the API endpoints.
