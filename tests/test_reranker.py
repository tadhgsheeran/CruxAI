import numpy as np
import pytest

from app.retrieval.reranker import (
    rerank_results,
)


def make_results() -> list[dict]:
    return [
        {
            "text": (
                "Heel hooks use the heel to pull "
                "against a hold."
            ),
            "source": "heel_hooks.md",
            "chunk_id": "heel_hooks.md_0",
            "score": 0.70,
        },
        {
            "text": (
                "Rest and sleep are important "
                "for recovery."
            ),
            "source": "recovery.md",
            "chunk_id": "recovery.md_0",
            "score": 0.65,
        },
    ]


def test_reranker_orders_by_cross_encoder_score(
    monkeypatch,
):
    class FakeReranker:
        def predict(
            self,
            pairs,
            show_progress_bar=False,
        ):
            return np.asarray(
                [0.2, 0.9]
            )

    monkeypatch.setattr(
        "app.retrieval.reranker."
        "get_reranker",
        lambda: FakeReranker(),
    )

    results = rerank_results(
        query="How does recovery work?",
        results=make_results(),
        top_k=2,
    )

    assert (
        results[0]["source"]
        == "recovery.md"
    )

    assert results[0][
        "reranker_score"
    ] == pytest.approx(0.9)

    assert "retrieval_score" in results[0]


def test_reranker_respects_top_k(
    monkeypatch,
):
    class FakeReranker:
        def predict(
            self,
            pairs,
            show_progress_bar=False,
        ):
            return np.asarray(
                [0.2, 0.9]
            )

    monkeypatch.setattr(
        "app.retrieval.reranker."
        "get_reranker",
        lambda: FakeReranker(),
    )

    results = rerank_results(
        query="How does recovery work?",
        results=make_results(),
        top_k=1,
    )

    assert len(results) == 1


def test_reranker_rejects_empty_query():
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        rerank_results(
            query=" ",
            results=make_results(),
        )


def test_reranker_rejects_invalid_top_k():
    with pytest.raises(
        ValueError,
        match="greater than 0",
    ):
        rerank_results(
            query="Test query",
            results=make_results(),
            top_k=0,
        )


def test_reranker_handles_empty_results():
    results = rerank_results(
        query="Test query",
        results=[],
        top_k=3,
    )

    assert results == []