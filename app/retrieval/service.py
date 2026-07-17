from ingestion.chunking import chunk_documents
from ingestion.embeddings import embed_chunks, load_embedding_model


class RetrievalService:
    def __init__(self):
        self.model = load_embedding_model()

        chunks = chunk_documents()
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
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

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
            seen_sources.add(result["source"])

            if len(diverse_results) == top_k:
                break

        return diverse_results

retrieval_service = RetrievalService()


if __name__ == "__main__":
    results = retrieval_service.search(
        "How can I keep my feet on the wall while climbing an overhang?",
        top_k=3,
    )

    for result in results:
        print(f"Source: {result['source']}")
        print(f"Score: {result['score']:.4f}")
        print(result["text"])
        print("-" * 60)