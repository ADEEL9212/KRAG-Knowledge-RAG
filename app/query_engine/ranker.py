"""
Document ranker for re-ranking search results.
"""

from typing import List, Dict
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentRanker:
    """Ranker for re-ranking retrieved documents."""

    def __init__(self, strategy: str = "similarity"):
        """
        Initialize the document ranker.

        Args:
            strategy: Ranking strategy ('similarity', 'diversity', 'mmr')
        """
        self.strategy = strategy
        logger.info(f"Initialized DocumentRanker with strategy={strategy}")

    def rank(
        self,
        documents: List[Dict],
        query: str = None,
        top_k: int = None,
    ) -> List[Dict]:
        """
        Rank or re-rank documents.

        Args:
            documents: List of documents to rank
            query: Original query (for context-aware ranking)
            top_k: Number of top results to return

        Returns:
            Ranked list of documents
        """
        if not documents:
            return []

        if self.strategy == "similarity":
            ranked = self._rank_by_similarity(documents)
        elif self.strategy == "diversity":
            ranked = self._rank_by_diversity(documents)
        elif self.strategy == "mmr":
            ranked = self._rank_by_mmr(documents)
        else:
            logger.warning(f"Unknown strategy {self.strategy}, using similarity")
            ranked = self._rank_by_similarity(documents)

        # Apply top_k filtering if specified
        if top_k is not None and top_k < len(ranked):
            ranked = ranked[:top_k]

        logger.info(f"Ranked {len(documents)} documents, returning {len(ranked)}")
        return ranked

    def _rank_by_similarity(self, documents: List[Dict]) -> List[Dict]:
        """
        Rank documents by similarity score (default behavior).

        Args:
            documents: List of documents with scores

        Returns:
            Sorted list of documents
        """
        # Already sorted by score from vector store
        return sorted(documents, key=lambda x: x.get("score", 0.0), reverse=True)

    def _rank_by_diversity(self, documents: List[Dict]) -> List[Dict]:
        """
        Rank documents to maximize diversity.

        This is a simplified implementation that alternates between
        high similarity and medium similarity results.

        Args:
            documents: List of documents

        Returns:
            Diversified list of documents
        """
        if len(documents) <= 2:
            return documents

        # Sort by score first
        sorted_docs = sorted(
            documents, key=lambda x: x.get("score", 0.0), reverse=True
        )

        # Interleave high-scoring and medium-scoring documents
        diverse_docs = []
        high_scores = sorted_docs[: len(sorted_docs) // 2]
        medium_scores = sorted_docs[len(sorted_docs) // 2 :]

        for i in range(max(len(high_scores), len(medium_scores))):
            if i < len(high_scores):
                diverse_docs.append(high_scores[i])
            if i < len(medium_scores):
                diverse_docs.append(medium_scores[i])

        return diverse_docs

    def _rank_by_mmr(
        self, documents: List[Dict], lambda_param: float = 0.5
    ) -> List[Dict]:
        """
        Maximal Marginal Relevance ranking.

        Balances relevance and diversity in results.

        Args:
            documents: List of documents
            lambda_param: Balance parameter (0=diversity, 1=relevance)

        Returns:
            MMR-ranked documents
        """
        if len(documents) <= 1:
            return documents

        # This is a simplified implementation
        # Full MMR would require embedding similarity calculations
        sorted_docs = sorted(
            documents, key=lambda x: x.get("score", 0.0), reverse=True
        )

        selected = [sorted_docs[0]]
        remaining = sorted_docs[1:]

        while remaining and len(selected) < len(documents):
            # Simple heuristic: pick documents with good scores
            # that are not too similar to already selected ones
            scores = []
            for doc in remaining:
                relevance_score = doc.get("score", 0.0)
                # Penalize if content is very similar to selected docs
                similarity_penalty = 0
                doc_content = doc.get("content", "").lower()
                for selected_doc in selected:
                    selected_content = selected_doc.get("content", "").lower()
                    # Simple word overlap as similarity measure
                    doc_words = set(doc_content.split())
                    selected_words = set(selected_content.split())
                    if doc_words and selected_words:
                        overlap = len(doc_words & selected_words) / len(
                            doc_words | selected_words
                        )
                        similarity_penalty = max(similarity_penalty, overlap)

                # MMR score
                mmr_score = (
                    lambda_param * relevance_score
                    - (1 - lambda_param) * similarity_penalty
                )
                scores.append((doc, mmr_score))

            # Select best scoring document
            best_doc = max(scores, key=lambda x: x[1])[0]
            selected.append(best_doc)
            remaining.remove(best_doc)

        return selected

    def filter_by_threshold(
        self, documents: List[Dict], threshold: float
    ) -> List[Dict]:
        """
        Filter documents by minimum similarity threshold.

        Args:
            documents: List of documents with scores
            threshold: Minimum score threshold

        Returns:
            Filtered list of documents
        """
        filtered = [doc for doc in documents if doc.get("score", 0.0) >= threshold]
        logger.info(
            f"Filtered {len(documents)} documents to {len(filtered)} "
            f"with threshold {threshold}"
        )
        return filtered
