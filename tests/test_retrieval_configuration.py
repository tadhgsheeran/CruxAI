import pytest

from app.retrieval.service import (
    RetrievalService,
)


def test_retrieval_service_stores_configuration():
    service = RetrievalService(
        chunk_size=250,
        overlap_paragraphs=0,
    )

    assert service.chunk_size == 250
    assert service.overlap_paragraphs == 0
    assert service.embedded_chunks


def test_retrieval_service_rejects_invalid_chunk_size():
    with pytest.raises(
        ValueError,
        match="chunk_size",
    ):
        RetrievalService(
            chunk_size=0,
        )


def test_retrieval_service_rejects_negative_overlap():
    with pytest.raises(
        ValueError,
        match="overlap_paragraphs",
    ):
        RetrievalService(
            overlap_paragraphs=-1,
        )


def test_configured_service_can_search():
    service = RetrievalService(
        chunk_size=250,
        overlap_paragraphs=1,
    )

    results = service.search(
        query="How do heel hooks work?",
        top_k=3,
    )

    assert len(results) == 3

    for result in results:
        assert result["source"]
        assert result["text"]
        assert result["chunk_id"]
        assert isinstance(
            result["score"],
            float,
        )