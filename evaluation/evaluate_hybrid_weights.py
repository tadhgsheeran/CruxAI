import json
import time
from pathlib import Path

from app.retrieval.service import RetrievalService
from evaluation.evaluate_retrieval import (
    evaluate_retrieval,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "hybrid_weight_ablation.json"
)


def main() -> None:
    dense_weights = [
        0.25,
        0.50,
        0.75,
        0.90,
    ]

    chunk_size = 500
    overlap_paragraphs = 2
    top_k = 3

    results = []

    for dense_weight in dense_weights:
        service = RetrievalService(
            chunk_size=chunk_size,
            overlap_paragraphs=overlap_paragraphs,
            retrieval_method="hybrid",
            hybrid_dense_weight=dense_weight,
        )

        evaluation_start = time.perf_counter()

        metrics = evaluate_retrieval(
            top_k=top_k,
            verbose=False,
            service=service,
        )

        evaluation_seconds = (
            time.perf_counter()
            - evaluation_start
        )

        result = {
            "dense_weight": dense_weight,
            "keyword_weight": 1.0 - dense_weight,
            "hit_rate": metrics["hit_rate"],
            "precision_at_k": (
                metrics["precision_at_k"]
            ),
            "recall_at_k": metrics["recall_at_k"],
            "mean_reciprocal_rank": (
                metrics["mean_reciprocal_rank"]
            ),
            "failure_count": (
                metrics["failure_count"]
            ),
            "evaluation_seconds": (
                evaluation_seconds
            ),
        }

        results.append(result)

        print(
            f"dense_weight={dense_weight:.2f} | "
            f"keyword_weight={1.0 - dense_weight:.2f} | "
            f"hit_rate={result['hit_rate']:.3f} | "
            f"precision={result['precision_at_k']:.3f} | "
            f"recall={result['recall_at_k']:.3f} | "
            f"MRR={result['mean_reciprocal_rank']:.3f}"
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
                "experiment": "hybrid_weight_ablation",
                "chunk_size": chunk_size,
                "overlap_paragraphs": (
                    overlap_paragraphs
                ),
                "top_k": top_k,
                "results": results,
            },
            file,
            indent=2,
        )

    print()
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()