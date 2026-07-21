from ingestion.chunking import chunk_documents
from ingestion.embeddings import (
    embed_chunks,
    load_embedding_model,
)


class RetrievalService:
    def __init__(
        self,
        chunk_size: int = 500,
        overlap_paragraphs: int = 1,
    ):
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0."
            )

        if overlap_paragraphs < 0:
            raise ValueError(
                "overlap_paragraphs cannot be negative."
            )

        self.chunk_size = chunk_size
        self.overlap_paragraphs = (
            overlap_paragraphs
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

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        scored_chunks = []

        for chunk in self.embedded_chunks:
            score = sum(
                query_value * chunk_value
                for query_value, chunk_value in zip(
                    query_embedding,
                    chunk["embedding"],
                )
            )

            scored_chunks.append(
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "score": float(score),
                }
            )

        scored_chunks.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

        diverse_results = []
        seen_sources = set()

        for result in scored_chunks:
            if result["source"] in seen_sources:
                continue

            diverse_results.append(result)
            seen_sources.add(
                result["source"]
            )

            if len(diverse_results) == top_k:
                break

        return diverse_results


retrieval_service = RetrievalService()