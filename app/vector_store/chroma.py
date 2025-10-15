"""
ChromaDB vector store implementation.
"""

import uuid
from typing import List, Dict, Optional
import numpy as np
import chromadb
from chromadb.config import Settings

from app.vector_store.base import VectorStore
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of vector store."""

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage (None for in-memory)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        logger.info(f"Initializing ChromaDB with collection: {collection_name}")

        try:
            if persist_directory:
                self.client = chromadb.Client(
                    Settings(
                        persist_directory=persist_directory,
                        anonymized_telemetry=False,
                    )
                )
                logger.info(f"Using persistent storage at: {persist_directory}")
            else:
                self.client = chromadb.Client(
                    Settings(anonymized_telemetry=False)
                )
                logger.info("Using in-memory storage")

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                f"ChromaDB initialized successfully. "
                f"Collection size: {self.collection.count()}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to ChromaDB collection.

        Args:
            texts: List of text content
            embeddings: Array of embeddings
            metadatas: Optional metadata for each document
            ids: Optional document IDs

        Returns:
            List of document IDs
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        # Prepare metadatas
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]

        # Ensure all metadata values are strings, ints, or floats
        clean_metadatas = []
        for metadata in metadatas:
            clean_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                else:
                    clean_metadata[key] = str(value)
            clean_metadatas.append(clean_metadata)

        try:
            # Convert embeddings to list format
            embeddings_list = embeddings.tolist()

            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=texts,
                metadatas=clean_metadatas,
            )

            logger.info(f"Added {len(texts)} documents to ChromaDB")
            return ids

        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {str(e)}")
            raise

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Search for similar documents in ChromaDB.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of search results
        """
        try:
            # Convert embedding to list
            query_embedding_list = query_embedding.tolist()

            # Prepare where clause for filtering
            where_clause = filter_metadata if filter_metadata else None

            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=top_k,
                where=where_clause,
            )

            # Format results
            formatted_results = []
            if results["documents"] and len(results["documents"]) > 0:
                for idx in range(len(results["documents"][0])):
                    result = {
                        "content": results["documents"][0][idx],
                        "metadata": results["metadatas"][0][idx]
                        if results["metadatas"]
                        else {},
                        "score": 1 - results["distances"][0][idx]
                        if results["distances"]
                        else 0.0,  # Convert distance to similarity
                        "id": results["ids"][0][idx],
                    }
                    formatted_results.append(result)

            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching ChromaDB: {str(e)}")
            raise

    def delete(self, ids: List[str]) -> bool:
        """
        Delete documents from ChromaDB.

        Args:
            ids: List of document IDs to delete

        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False

    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}

    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.

        Returns:
            True if successful
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Cleared collection: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False
