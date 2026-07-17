from pathlib import Path


DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"


def load_markdown_documents() -> list[dict]:
    documents = []

    for file_path in sorted(DOCUMENTS_DIR.glob("*.md")):
        text = file_path.read_text(encoding="utf-8").strip()

        documents.append(
            {
                "text": text,
                "source": file_path.name,
                "path": str(file_path),
            }
        )

    return documents


if __name__ == "__main__":
    loaded_documents = load_markdown_documents()

    print(f"Loaded {len(loaded_documents)} documents.\n")

    for document in loaded_documents:
        print(f"Source: {document['source']}")
        print(f"Characters: {len(document['text'])}")
        print("-" * 40)