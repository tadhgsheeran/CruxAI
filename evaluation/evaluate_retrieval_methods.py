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
    / "retrieval_method_ablation.json"
)


def main() -> None:
    retrieval_methods = [
        "dense",
        "keyword",
        "hybrid",
    ]

    chunk_size = 500
    overlap_paragraphs = 2
    top_k = 3

    results = []

    for retrieval_method in retrieval_methods:
        build_start = time.perf_counter()

        service = RetrievalService(
            chunk_size=chunk_size,
            overlap_paragraphs=(
                overlap_paragraphs
            ),
            retrieval_method=(
                retrieval_method
            ),
            hybrid_dense_weight=0.5,
        )

        index_build_seconds = (
            time.perf_counter() - build_start
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
            "retrieval_method": (
                retrieval_method
            ),
            "chunk_size": chunk_size,
            "overlap_paragraphs": (
                overlap_paragraphs
            ),
            "top_k": top_k,
            "hybrid_dense_weight": (
                0.5
                if retrieval_method == "hybrid"
                else None
            ),
            "chunk_count": len(
                service.embedded_chunks
            ),
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
            "index_build_seconds": (
                index_build_seconds
            ),
            "evaluation_seconds": (
                evaluation_seconds
            ),
        }

        results.append(result)

        print(
            f"method={retrieval_method:7} | "
            f"hit_rate="
            f"{result['hit_rate']:.3f} | "
            f"precision="
            f"{result['precision_at_k']:.3f} | "
            f"recall="
            f"{result['recall_at_k']:.3f} | "
            f"MRR="
            f"{result['mean_reciprocal_rank']:.3f} | "
            f"evaluation="
            f"{result['evaluation_seconds']:.2f}s"
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
                    "retrieval_method_ablation"
                ),
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
    print(
        f"Saved results to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()