"""
Answer synthesizer using LLM for generating responses from retrieved documents.
"""

from typing import List, Dict, Optional
import openai

from app.utils.logger import get_logger

logger = get_logger(__name__)


class AnswerSynthesizer:
    """Synthesizer for generating answers using LLM and retrieved context."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500,
    ):
        """
        Initialize the answer synthesizer.

        Args:
            api_key: OpenAI API key
            model: Model name to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not api_key:
            logger.warning("No OpenAI API key provided. Synthesis will be disabled.")
            self.client = None
        else:
            self.client = openai.OpenAI(api_key=api_key)
            logger.info(f"Initialized AnswerSynthesizer with model={model}")

    def synthesize(
        self,
        query: str,
        documents: List[Dict],
        include_sources: bool = True,
    ) -> Dict:
        """
        Generate an answer based on query and retrieved documents.

        Args:
            query: User query
            documents: Retrieved documents
            include_sources: Whether to include source information

        Returns:
            Dictionary with answer and metadata
        """
        if not self.client:
            logger.warning("OpenAI client not initialized. Returning concatenated context.")
            return self._fallback_response(query, documents)

        if not documents:
            return {
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
                "model": self.model,
            }

        try:
            # Build context from documents
            context = self._build_context(documents)

            # Create prompt
            prompt = self._create_prompt(query, context)

            logger.debug(f"Generating answer for query: {query[:100]}...")

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that answers questions "
                            "based on the provided context. Always cite your sources "
                            "when providing information. If the context doesn't contain "
                            "enough information to answer the question, say so."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content

            result = {
                "answer": answer,
                "model": self.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }

            if include_sources:
                result["sources"] = self._format_sources(documents)

            logger.info("Successfully generated answer")
            return result

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return self._fallback_response(query, documents)

    def _build_context(self, documents: List[Dict]) -> str:
        """
        Build context string from documents.

        Args:
            documents: List of retrieved documents

        Returns:
            Context string
        """
        context_parts = []
        for idx, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            source = metadata.get("filename", "Unknown")

            context_parts.append(f"[Source {idx}: {source}]\n{content}\n")

        return "\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for LLM.

        Args:
            query: User query
            context: Context from retrieved documents

        Returns:
            Formatted prompt
        """
        prompt = f"""Context information from the knowledge base:

{context}

Based on the context above, please answer the following question. 
If the context doesn't contain relevant information, please state that clearly.

Question: {query}

Answer:"""
        return prompt

    def _format_sources(self, documents: List[Dict]) -> List[Dict]:
        """
        Format source information from documents.

        Args:
            documents: List of documents

        Returns:
            List of formatted sources
        """
        sources = []
        for doc in documents:
            source = {
                "content": doc.get("content", "")[:200] + "...",  # Truncate
                "metadata": doc.get("metadata", {}),
                "score": doc.get("score", 0.0),
            }
            sources.append(source)
        return sources

    def _fallback_response(self, query: str, documents: List[Dict]) -> Dict:
        """
        Generate fallback response when LLM is not available.

        Args:
            query: User query
            documents: Retrieved documents

        Returns:
            Response dictionary
        """
        if not documents:
            answer = "No relevant documents found to answer this question."
        else:
            # Concatenate top document contents
            top_contents = [
                doc.get("content", "")[:500] for doc in documents[:3]
            ]
            answer = (
                "Based on the retrieved documents:\n\n"
                + "\n\n".join(top_contents)
            )

        return {
            "answer": answer,
            "sources": self._format_sources(documents),
            "model": "fallback",
        }

    def synthesize_streaming(
        self, query: str, documents: List[Dict]
    ):
        """
        Generate answer with streaming response (for future implementation).

        Args:
            query: User query
            documents: Retrieved documents

        Yields:
            Answer chunks
        """
        if not self.client:
            logger.warning("Streaming not available without OpenAI client")
            yield self._fallback_response(query, documents)["answer"]
            return

        try:
            context = self._build_context(documents)
            prompt = self._create_prompt(query, context)

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that answers questions "
                            "based on the provided context."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error in streaming synthesis: {str(e)}")
            yield "Error generating response."
