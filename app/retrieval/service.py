import math
import re
from collections import Counter

from ingestion.chunking import chunk_documents
from ingestion.embeddings import (
    embed_chunks,
    load_embedding_model,
)

from app.retrieval.reranker import (
    rerank_results,
)

SUPPORTED_RETRIEVAL_METHODS = {
    "dense",
    "keyword",
    "hybrid",
}


def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase searchable tokens.
    """
    return re.findall(
        r"[a-z0-9]+(?:-[a-z0-9]+)?",
        text.lower(),
    )


class RetrievalService:
    def __init__(
        self,
        chunk_size: int = 500,
        overlap_paragraphs: int = 2,
        retrieval_method: str = "dense",
        hybrid_dense_weight: float = 0.5,
    ):
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0."
            )

        if overlap_paragraphs < 0:
            raise ValueError(
                "overlap_paragraphs cannot be negative."
            )

        if (
            retrieval_method
            not in SUPPORTED_RETRIEVAL_METHODS
        ):
            raise ValueError(
                "retrieval_method must be one of: "
                "dense, keyword, hybrid."
            )

        if not 0.0 <= hybrid_dense_weight <= 1.0:
            raise ValueError(
                "hybrid_dense_weight must be between "
                "0 and 1."
            )

        self.chunk_size = chunk_size
        self.overlap_paragraphs = (
            overlap_paragraphs
        )
        self.retrieval_method = retrieval_method
        self.hybrid_dense_weight = (
            hybrid_dense_weight
        )

        self.model = load_embedding_model()

        chunks = chunk_documents(
            chunk_size=chunk_size,
            overlap_paragraphs=(
                overlap_paragraphs
            ),
        )

        self.embedded_chunks = embed_chunks(
            chunks,
            self.model,
        )

        self.document_frequencies = (
            self._calculate_document_frequencies()
        )

        self.keyword_vectors = [
            self._build_keyword_vector(
                chunk["text"]
            )
            for chunk in self.embedded_chunks
        ]

    def _calculate_document_frequencies(
        self,
    ) -> Counter:
        """
        Count how many chunks contain each token.
        """
        document_frequencies = Counter()

        for chunk in self.embedded_chunks:
            unique_tokens = set(
                tokenize(chunk["text"])
            )

            document_frequencies.update(
                unique_tokens
            )

        return document_frequencies

    def _inverse_document_frequency(
        self,
        token: str,
    ) -> float:
        """
        Calculate smoothed inverse document frequency.
        """
        total_documents = len(
            self.embedded_chunks
        )

        document_frequency = (
            self.document_frequencies.get(
                token,
                0,
            )
        )

        return (
            math.log(
                (total_documents + 1)
                / (document_frequency + 1)
            )
            + 1.0
        )

    def _build_keyword_vector(
        self,
        text: str,
    ) -> dict[str, float]:
        """
        Build a normalized TF-IDF vector.
        """
        tokens = tokenize(text)

        if not tokens:
            return {}

        token_counts = Counter(tokens)
        total_tokens = len(tokens)

        vector = {}

        for token, count in token_counts.items():
            term_frequency = (
                count / total_tokens
            )

            inverse_document_frequency = (
                self._inverse_document_frequency(
                    token
                )
            )

            vector[token] = (
                term_frequency
                * inverse_document_frequency
            )

        magnitude = math.sqrt(
            sum(
                value ** 2
                for value in vector.values()
            )
        )

        if magnitude == 0:
            return {}

        return {
            token: value / magnitude
            for token, value in vector.items()
        }

    @staticmethod
    def _keyword_cosine_similarity(
        query_vector: dict[str, float],
        document_vector: dict[str, float],
    ) -> float:
        """
        Calculate cosine similarity between sparse vectors.
        """
        if not query_vector or not document_vector:
            return 0.0

        shared_tokens = (
            query_vector.keys()
            & document_vector.keys()
        )

        return float(
            sum(
                query_vector[token]
                * document_vector[token]
                for token in shared_tokens
            )
        )

    def _dense_scores(
        self,
        query: str,
    ) -> list[float]:
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        scores = []

        for chunk in self.embedded_chunks:
            score = sum(
                query_value * chunk_value
                for query_value, chunk_value in zip(
                    query_embedding,
                    chunk["embedding"],
                )
            )

            scores.append(float(score))

        return scores

    def _keyword_scores(
        self,
        query: str,
    ) -> list[float]:
        query_vector = (
            self._build_keyword_vector(query)
        )

        return [
            self._keyword_cosine_similarity(
                query_vector=query_vector,
                document_vector=document_vector,
            )
            for document_vector in self.keyword_vectors
        ]

    @staticmethod
    def _normalize_scores(
        scores: list[float],
    ) -> list[float]:
        """
        Min-max normalize scores to the range 0–1.
        """
        if not scores:
            return []

        minimum = min(scores)
        maximum = max(scores)

        if maximum == minimum:
            return [0.0 for _ in scores]

        return [
            (score - minimum)
            / (maximum - minimum)
            for score in scores
        ]

    def search(
        self,
        query: str,
        top_k: int = 3,
        retrieval_method: str | None = None,
        rerank: bool = False,
        candidate_k: int = 8,
    ) -> list[dict]:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than 0."
            )

        if rerank and candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or "
                "equal to top_k when reranking."
            )
            
        active_method = (
            retrieval_method
            if retrieval_method is not None
            else self.retrieval_method
        )

        if (
            active_method
            not in SUPPORTED_RETRIEVAL_METHODS
        ):
            raise ValueError(
                "retrieval_method must be one of: "
                "dense, keyword, hybrid."
            )

        dense_scores = None
        keyword_scores = None

        if active_method in {
            "dense",
            "hybrid",
        }:
            dense_scores = self._dense_scores(
                query
            )

        if active_method in {
            "keyword",
            "hybrid",
        }:
            keyword_scores = (
                self._keyword_scores(query)
            )

        if active_method == "dense":
            final_scores = dense_scores

        elif active_method == "keyword":
            final_scores = keyword_scores

        else:
            normalized_dense = (
                self._normalize_scores(
                    dense_scores
                )
            )

            normalized_keyword = (
                self._normalize_scores(
                    keyword_scores
                )
            )

            dense_weight = (
                self.hybrid_dense_weight
            )

            keyword_weight = (
                1.0 - dense_weight
            )

            final_scores = [
                (
                    dense_weight
                    * dense_score
                )
                + (
                    keyword_weight
                    * keyword_score
                )
                for dense_score, keyword_score
                in zip(
                    normalized_dense,
                    normalized_keyword,
                )
            ]

        scored_chunks = []

        for index, chunk in enumerate(
            self.embedded_chunks
        ):
            scored_chunks.append(
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "score": float(
                        final_scores[index]
                    ),
                    "retrieval_method": (
                        active_method
                    ),
                    "dense_score": (
                        float(dense_scores[index])
                        if dense_scores is not None
                        else None
                    ),
                    "keyword_score": (
                        float(
                            keyword_scores[index]
                        )
                        if keyword_scores
                        is not None
                        else None
                    ),
                }
            )

        scored_chunks.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

        diverse_results = []
        seen_sources = set()

        result_limit = (
            candidate_k
            if rerank
            else top_k
        )

        for result in scored_chunks:
            if result["source"] in seen_sources:
                continue

            diverse_results.append(result)

            seen_sources.add(
                result["source"]
            )

            if len(diverse_results) == result_limit:
                break

        if rerank:
            return rerank_results(
                query=query,
                results=diverse_results,
                top_k=top_k,
            )

        return diverse_results


retrieval_service = RetrievalService(
    chunk_size=500,
    overlap_paragraphs=2,
    retrieval_method="hybrid",
    hybrid_dense_weight=0.90,
)