from ingestion.loaders import load_markdown_documents


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap_paragraphs: int = 1,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap_paragraphs < 0:
        raise ValueError("overlap_paragraphs cannot be negative.")

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_paragraphs = []
    current_length = 0

    for paragraph in paragraphs:
        added_length = len(paragraph)

        if current_paragraphs:
            added_length += 2

        if (
            current_paragraphs
            and current_length + added_length > chunk_size
        ):
            chunks.append("\n\n".join(current_paragraphs))

            if overlap_paragraphs > 0:
                current_paragraphs = current_paragraphs[
                    -overlap_paragraphs:
                ]
                current_length = len(
                    "\n\n".join(current_paragraphs)
                )
            else:
                current_paragraphs = []
                current_length = 0

        current_paragraphs.append(paragraph)
        current_length = len(
            "\n\n".join(current_paragraphs)
        )

    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs))

    return chunks


def chunk_documents(
    chunk_size: int = 500,
    overlap_paragraphs: int = 1,
) -> list[dict]:
    documents = load_markdown_documents()
    all_chunks = []

    for document in documents:
        text_chunks = split_text_into_chunks(
            document["text"],
            chunk_size=chunk_size,
            overlap_paragraphs=overlap_paragraphs,
        )

        for index, chunk_text in enumerate(text_chunks):
            all_chunks.append(
                {
                    "text": chunk_text,
                    "source": document["source"],
                    "path": document["path"],
                    "chunk_id": f"{document['source']}_{index}",
                    "chunk_index": index,
                }
            )

    return all_chunks


if __name__ == "__main__":
    chunks = chunk_documents()

    print(f"Created {len(chunks)} chunks.\n")

    for chunk in chunks:
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Characters: {len(chunk['text'])}")
        print(chunk["text"])
        print("-" * 60)