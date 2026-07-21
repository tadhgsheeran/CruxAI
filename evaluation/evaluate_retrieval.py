import json
from pathlib import Path

from app.retrieval.service import (
    RetrievalService,
    retrieval_service,
)

BENCHMARK_PATH = (
    Path(__file__).resolve().parent
    / "retrieval_benchmark.jsonl"
)


def load_benchmark() -> list[dict]:
    examples = []

    with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                examples.append(json.loads(line))

    return examples


def evaluate_retrieval(
    top_k: int = 3,
    verbose: bool = True,
    service: RetrievalService | None = None,
) -> dict:

    examples = load_benchmark()

    active_service = (
        service
        if service is not None
        else retrieval_service
    )
    
    hits = 0
    reciprocal_ranks = []

    precision_scores = []
    recall_scores = []

    failures = []

    for example in examples:
        question = example["question"]
        relevant_sources = set(
            example["relevant_sources"]
        )

        results = active_service.search(
            query=question,
            top_k=top_k,
        )

        retrieved_sources = [
            result["source"]
            for result in results
        ]

        retrieved_source_set = set(
            retrieved_sources
        )

        relevant_retrieved = (
            retrieved_source_set
            & relevant_sources
        )

        precision_at_k = (
            len(relevant_retrieved)
            / len(retrieved_sources)
            if retrieved_sources
            else 0.0
        )

        recall_at_k = (
            len(relevant_retrieved)
            / len(relevant_sources)
            if relevant_sources
            else 0.0
        )

        precision_scores.append(
            precision_at_k
        )

        recall_scores.append(
            recall_at_k
        )

        first_relevant_rank = None

        for rank, source in enumerate(
            retrieved_sources,
            start=1,
        ):
            if source in relevant_sources:
                first_relevant_rank = rank
                break

        if first_relevant_rank is not None:
            hits += 1

            reciprocal_ranks.append(
                1 / first_relevant_rank
            )
        else:
            reciprocal_ranks.append(0.0)

            failures.append(
                {
                    "question": question,
                    "relevant_sources": sorted(
                        relevant_sources
                    ),
                    "retrieved_sources": (
                        retrieved_sources
                    ),
                }
            )

        if verbose:
            print(f"Question: {question}")
            print(
                "Expected:",
                sorted(relevant_sources),
            )
            print(
                "Retrieved:",
                retrieved_sources,
            )
            print(
                "Precision@k:",
                f"{precision_at_k:.3f}",
            )
            print(
                "Recall@k:",
                f"{recall_at_k:.3f}",
            )
            print(
                "First relevant rank:",
                first_relevant_rank,
            )
            print("-" * 60)

    total = len(examples)

    hit_rate = (
        hits / total
        if total
        else 0.0
    )

    mean_reciprocal_rank = (
        sum(reciprocal_ranks) / total
        if total
        else 0.0
    )

    mean_precision_at_k = (
        sum(precision_scores) / total
        if total
        else 0.0
    )

    mean_recall_at_k = (
        sum(recall_scores) / total
        if total
        else 0.0
    )

    return {
        "questions": total,
        "top_k": top_k,
        "hits": hits,
        "hit_rate": hit_rate,
        "precision_at_k": (
            mean_precision_at_k
        ),
        "recall_at_k": mean_recall_at_k,
        "mean_reciprocal_rank": (
            mean_reciprocal_rank
        ),
        "failure_count": len(failures),
        "failures": failures,
    }

if __name__ == "__main__":
    metrics = evaluate_retrieval(
        top_k=3,
        verbose=True,
    )

    print("\nRetrieval metrics")
    print(f"Questions: {metrics['questions']}")
    print(f"Top-k: {metrics['top_k']}")
    print(
        "Hit rate:",
        f"{metrics['hit_rate']:.3f}",
    )
    print(
        "Precision@k:",
        f"{metrics['precision_at_k']:.3f}",
    )
    print(
        "Recall@k:",
        f"{metrics['recall_at_k']:.3f}",
    )
    print(
        "Mean reciprocal rank:",
        f"{metrics['mean_reciprocal_rank']:.3f}",
    )