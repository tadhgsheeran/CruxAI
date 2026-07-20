import json
from pathlib import Path

from app.retrieval.service import retrieval_service


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


def evaluate_retrieval(top_k: int = 3) -> dict:
    examples = load_benchmark()

    hits = 0
    reciprocal_ranks = []

    for example in examples:
        question = example["question"]
        relevant_sources = set(example["relevant_sources"])

        results = retrieval_service.search(
            query=question,
            top_k=top_k,
        )

        retrieved_sources = [
            result["source"]
            for result in results
        ]

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
            reciprocal_ranks.append(0)

        print(f"Question: {question}")
        print(f"Expected: {sorted(relevant_sources)}")
        print(f"Retrieved: {retrieved_sources}")
        print(
            "First relevant rank:",
            first_relevant_rank,
        )
        print("-" * 60)

    total = len(examples)

    hit_rate = hits / total if total else 0
    mean_reciprocal_rank = (
        sum(reciprocal_ranks) / total
        if total
        else 0
    )

    return {
        "questions": total,
        "top_k": top_k,
        "hit_rate": hit_rate,
        "mean_reciprocal_rank": mean_reciprocal_rank,
    }


if __name__ == "__main__":
    metrics = evaluate_retrieval(top_k=3)

    print("\nRetrieval metrics")
    print(f"Questions: {metrics['questions']}")
    print(f"Top-k: {metrics['top_k']}")
    print(f"Hit rate: {metrics['hit_rate']:.3f}")
    print(
        "Mean reciprocal rank:",
        f"{metrics['mean_reciprocal_rank']:.3f}",
    )