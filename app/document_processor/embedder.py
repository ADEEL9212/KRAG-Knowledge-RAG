"""
Document embedder for generating embeddings using Sentence Transformers.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentEmbedder:
    """Embedder for generating vector embeddings from text."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the document embedder.

        Args:
            model_name: Name of the Sentence Transformer model to use
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")

        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Loaded model {model_name} with dimension {self.embedding_dimension}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text or list of texts.

        Args:
            text: Single text string or list of text strings

        Returns:
            Numpy array of embeddings (2D array for batch input)
        """
        if isinstance(text, str):
            text = [text]

        if not text or len(text) == 0:
            logger.warning("Empty text provided for embedding")
            return np.array([])

        try:
            embeddings = self.model.encode(
                text, convert_to_numpy=True, show_progress_bar=False
            )
            logger.debug(f"Generated embeddings for {len(text)} texts")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def embed_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for a large batch of texts efficiently.

        Args:
            texts: List of text strings
            batch_size: Size of batches for processing

        Returns:
            Numpy array of embeddings
        """
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return np.array([])

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100,
            )
            logger.info(f"Generated embeddings for {len(texts)} texts in batches")
            return embeddings

        except Exception as e:
            logger.error(f"Error in batch embedding: {str(e)}")
            raise

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.

        Returns:
            Embedding dimension
        """
        return self.embedding_dimension
