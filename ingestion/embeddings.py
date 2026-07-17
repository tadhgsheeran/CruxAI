from sentence_transformers import SentenceTransformer

from ingestion.chunking import chunk_documents


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
) -> list[dict]:
    if not chunks:
        return []

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append(
            {
                **chunk,
                "embedding": embedding.tolist(),
            }
        )

    return embedded_chunks


if __name__ == "__main__":
    chunks = chunk_documents()
    model = load_embedding_model()
    embedded_chunks = embed_chunks(chunks, model)

    print(f"Embedded {len(embedded_chunks)} chunks.\n")

    for chunk in embedded_chunks:
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Embedding dimensions: {len(chunk['embedding'])}")
        print(f"First five values: {chunk['embedding'][:5]}")
        print("-" * 60)