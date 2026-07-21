from functools import lru_cache

from sentence_transformers import CrossEncoder


RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """
    Load and cache the cross-encoder reranker.
    """
    return CrossEncoder(
        RERANKER_MODEL_NAME,
    )


def rerank_results(
    query: str,
    results: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    Rerank retrieved documents using a cross-encoder.
    """
    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    if not results:
        return []

    reranker = get_reranker()

    query_document_pairs = [
        [
            query,
            result["text"],
        ]
        for result in results
    ]

    reranker_scores = reranker.predict(
        query_document_pairs,
        show_progress_bar=False,
    )

    reranked_results = []

    for result, reranker_score in zip(
        results,
        reranker_scores,
    ):
        reranked_result = dict(result)

        reranked_result[
            "retrieval_score"
        ] = result["score"]

        reranked_result[
            "reranker_score"
        ] = float(reranker_score)

        reranked_result["score"] = float(
            reranker_score
        )

        reranked_results.append(
            reranked_result
        )

    reranked_results.sort(
        key=lambda result: result[
            "reranker_score"
        ],
        reverse=True,
    )

    return reranked_results[:top_k]