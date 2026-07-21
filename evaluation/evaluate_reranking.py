import json
import time
from pathlib import Path

from app.retrieval.service import (
    RetrievalService,
)
from evaluation.evaluate_retrieval import (
    evaluate_retrieval,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "reranking_ablation.json"
)


def main() -> None:
    chunk_size = 500
    overlap_paragraphs = 2
    top_k = 3
    candidate_k = 8

    service = RetrievalService(
        chunk_size=chunk_size,
        overlap_paragraphs=(
            overlap_paragraphs
        ),
        retrieval_method="hybrid",
        hybrid_dense_weight=0.90,
    )

    configurations = [
        {
            "name": "reranking_off",
            "rerank": False,
        },
        {
            "name": "reranking_on",
            "rerank": True,
        },
    ]

    results = []

    for configuration in configurations:
        start = time.perf_counter()

        metrics = evaluate_retrieval(
            top_k=top_k,
            verbose=False,
            service=service,
            rerank=configuration["rerank"],
            candidate_k=candidate_k,
        )

        evaluation_seconds = (
            time.perf_counter() - start
        )

        result = {
            "configuration": (
                configuration["name"]
            ),
            "rerank": configuration[
                "rerank"
            ],
            "candidate_k": (
                candidate_k
                if configuration["rerank"]
                else None
            ),
            "top_k": top_k,
            "hit_rate": metrics[
                "hit_rate"
            ],
            "precision_at_k": metrics[
                "precision_at_k"
            ],
            "recall_at_k": metrics[
                "recall_at_k"
            ],
            "mean_reciprocal_rank": metrics[
                "mean_reciprocal_rank"
            ],
            "failure_count": metrics[
                "failure_count"
            ],
            "evaluation_seconds": (
                evaluation_seconds
            ),
            "average_query_seconds": (
                evaluation_seconds
                / metrics["questions"]
                if metrics["questions"]
                else 0.0
            ),
        }

        results.append(result)

        print(
            f"{result['configuration']:14} | "
            f"hit_rate="
            f"{result['hit_rate']:.3f} | "
            f"precision="
            f"{result['precision_at_k']:.3f} | "
            f"recall="
            f"{result['recall_at_k']:.3f} | "
            f"MRR="
            f"{result['mean_reciprocal_rank']:.3f} | "
            f"total="
            f"{result['evaluation_seconds']:.2f}s | "
            f"per_query="
            f"{result['average_query_seconds']:.3f}s"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "experiment": (
                    "reranking_ablation"
                ),
                "retrieval_method": "hybrid",
                "hybrid_dense_weight": 0.90,
                "chunk_size": chunk_size,
                "overlap_paragraphs": (
                    overlap_paragraphs
                ),
                "top_k": top_k,
                "candidate_k": candidate_k,
                "reranker_model": (
                    "cross-encoder/"
                    "ms-marco-MiniLM-L-6-v2"
                ),
                "results": results,
            },
            file,
            indent=2,
        )

    print()
    print(
        f"Saved results to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()