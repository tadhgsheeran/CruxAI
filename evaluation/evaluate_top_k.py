import json
from pathlib import Path

from evaluation.evaluate_retrieval import (
    evaluate_retrieval,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "top_k_ablation.json"
)


def main() -> None:
    top_k_values = [1, 2, 3, 4, 5]

    results = []

    for top_k in top_k_values:
        metrics = evaluate_retrieval(
            top_k=top_k,
            verbose=False,
        )

        result = {
            "top_k": top_k,
            "hit_rate": metrics["hit_rate"],
            "precision_at_k": (
                metrics["precision_at_k"]
            ),
            "recall_at_k": (
                metrics["recall_at_k"]
            ),
            "mean_reciprocal_rank": (
                metrics["mean_reciprocal_rank"]
            ),
            "failure_count": (
                metrics["failure_count"]
            ),
        }

        results.append(result)

        print(
            f"top_k={top_k} | "
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
                "experiment": "top_k_ablation",
                "retrieval_method": "dense",
                "results": results,
            },
            file,
            indent=2,
        )

    print()
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()