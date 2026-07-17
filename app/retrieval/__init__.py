def __init__(self):
    self.model = load_embedding_model()

    chunks = chunk_documents()
    self.embedded_chunks = embed_chunks(
        chunks,
        self.model,
    )