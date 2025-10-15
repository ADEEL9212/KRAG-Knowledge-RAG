"""
Document retriever for searching the vector store.
"""

from typing import List, Dict, Optional
import numpy as np

from app.vector_store.base import VectorStore
from app.document_processor.embedder import DocumentEmbedder
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentRetriever:
    """Retriever for finding relevant documents using vector search."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: DocumentEmbedder,
        top_k: int = 5,
    ):
        """
        Initialize the document retriever.

        Args:
            vector_store: Vector store instance
            embedder: Document embedder instance
            top_k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

        logger.info(f"Initialized DocumentRetriever with top_k={top_k}")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of results to return (overrides default)
            filter_metadata: Optional metadata filters

        Returns:
            List of retrieved documents with scores
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        k = top_k if top_k is not None else self.top_k

        try:
            # Generate query embedding
            logger.debug(f"Generating embedding for query: {query[:100]}...")
            query_embedding = self.embedder.embed(query)

            # Handle both single and batch embedding returns
            if len(query_embedding.shape) > 1:
                query_embedding = query_embedding[0]

            # Search vector store
            logger.debug(f"Searching vector store for top {k} results")
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=k,
                filter_metadata=filter_metadata,
            )

            logger.info(f"Retrieved {len(results)} documents for query")
            return results

        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[List[Dict]]:
        """
        Retrieve documents for multiple queries.

        Args:
            queries: List of search queries
            top_k: Number of results per query

        Returns:
            List of result lists, one per query
        """
        results = []
        for query in queries:
            try:
                query_results = self.retrieve(query, top_k=top_k)
                results.append(query_results)
            except Exception as e:
                logger.error(f"Error retrieving for query '{query[:50]}': {str(e)}")
                results.append([])

        return results

    def get_stats(self) -> Dict:
        """
        Get retriever statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "top_k": self.top_k,
            "vector_store_stats": self.vector_store.get_collection_stats(),
        }
