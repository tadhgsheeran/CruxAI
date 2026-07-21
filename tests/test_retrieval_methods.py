import pytest

from app.retrieval.service import (
    RetrievalService,
    tokenize,
)


def test_tokenize_text():
    tokens = tokenize(
        "Heel-hooks help on steep routes."
    )

    assert "heel-hooks" in tokens
    assert "steep" in tokens
    assert "routes" in tokens


@pytest.mark.parametrize(
    "retrieval_method",
    [
        "dense",
        "keyword",
        "hybrid",
    ],
)
def test_supported_retrieval_methods(
    retrieval_method,
):
    service = RetrievalService(
        retrieval_method=retrieval_method,
    )

    results = service.search(
        query="How do heel hooks work?",
        top_k=3,
    )

    assert len(results) == 3

    for result in results:
        assert (
            result["retrieval_method"]
            == retrieval_method
        )


def test_rejects_invalid_retrieval_method():
    with pytest.raises(
        ValueError,
        match="retrieval_method",
    ):
        RetrievalService(
            retrieval_method="invalid",
        )


def test_hybrid_results_include_component_scores():
    service = RetrievalService(
        retrieval_method="hybrid",
    )

    results = service.search(
        query="How do I improve heel hooks?",
        top_k=3,
    )

    for result in results:
        assert result["dense_score"] is not None
        assert (
            result["keyword_score"]
            is not None
        )


def test_rejects_invalid_hybrid_weight():
    with pytest.raises(
        ValueError,
        match="hybrid_dense_weight",
    ):
        RetrievalService(
            retrieval_method="hybrid",
            hybrid_dense_weight=1.5,
        )
        
def test_search_can_rerank_results(
    monkeypatch,
):
    def fake_rerank_results(
        query,
        results,
        top_k,
    ):
        reranked = list(
            reversed(results)
        )

        return reranked[:top_k]

    monkeypatch.setattr(
        "app.retrieval.service."
        "rerank_results",
        fake_rerank_results,
    )

    service = RetrievalService(
        retrieval_method="hybrid",
        hybrid_dense_weight=0.90,
    )

    results = service.search(
        query="How do heel hooks work?",
        top_k=3,
        rerank=True,
        candidate_k=8,
    )

    assert len(results) == 3


def test_reranking_requires_enough_candidates():
    service = RetrievalService(
        retrieval_method="hybrid",
    )

    with pytest.raises(
        ValueError,
        match="candidate_k",
    ):
        service.search(
            query="How do heel hooks work?",
            top_k=5,
            rerank=True,
            candidate_k=3,
        )