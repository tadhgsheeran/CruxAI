import json
from pathlib import Path

from app.generation.service import generation_service
from app.retrieval.service import retrieval_service

MIN_RETRIEVAL_SCORE = 0.35

BENCHMARK_PATH = Path("evaluation/rag_benchmark.jsonl")

REFUSAL_PHRASES = [
    "not enough information",
    "not have enough relevant information",
    "does not contain enough information",
    "cannot determine",
    "i don't know",
    "i do not know",
    "not provided",
    "not available in the context",
]


def load_benchmark() -> list[dict]:
    examples = []

    with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                examples.append(json.loads(line))

    return examples


def contains_expected_source(
    retrieved_sources: list[str],
    expected_sources: list[str],
) -> bool:
    return any(
        source in retrieved_sources
        for source in expected_sources
    )


def contains_expected_term(
    answer: str,
    expected_terms: list[str],
) -> bool:
    answer_lower = answer.lower()

    return any(
        term.lower() in answer_lower
        for term in expected_terms
    )


def contains_citation(
    answer: str,
    retrieved_sources: list[str],
) -> bool:
    return any(
        f"[{source}]" in answer
        for source in retrieved_sources
    )


def is_cautious_refusal(answer: str) -> bool:
    answer_lower = answer.lower()

    return any(
        phrase in answer_lower
        for phrase in REFUSAL_PHRASES
    )


def main() -> None:
    examples = load_benchmark()

    retrieval_passes = 0
    citation_passes = 0
    content_passes = 0
    refusal_passes = 0
    answerable_count = 0
    unanswerable_count = 0

    for index, example in enumerate(examples, start=1):
        question = example["question"]
        expected_sources = example["expected_sources"]
        expected_terms = example["expected_terms"]
        answerable = example["answerable"]

        results = retrieval_service.search(
            query=question,
            top_k=3,
        )

        if not results or results[0]["score"] < MIN_RETRIEVAL_SCORE:
            answer = (
                "I do not have enough relevant information in the "
                "CruxAI knowledge base to answer that question."
            )
            results = []
        else:
            answer = generation_service.generate_answer(
                query=question,
                retrieved_results=results,
            )

        retrieved_sources = [
            result["source"]
            for result in results
        ]
        
        retrieved_scores = [
            round(result["score"], 4)
            for result in results
        ]
        
        print("=" * 80)
        print(f"Question {index}: {question}")
        print(f"Answerable: {answerable}")
        print(f"Expected sources: {expected_sources}")
        print(f"Retrieved sources: {retrieved_sources}")
        print(f"Retrieved scores: {retrieved_scores}")
        print(f"Answer:\n{answer}")

        if answerable:
            answerable_count += 1

            retrieval_pass = contains_expected_source(
                retrieved_sources,
                expected_sources,
            )

            citation_pass = contains_citation(
                answer,
                retrieved_sources,
            )

            content_pass = contains_expected_term(
                answer,
                expected_terms,
            )

            retrieval_passes += int(retrieval_pass)
            citation_passes += int(citation_pass)
            content_passes += int(content_pass)

            print(f"Retrieval pass: {retrieval_pass}")
            print(f"Citation pass: {citation_pass}")
            print(f"Content pass: {content_pass}")

        else:
            unanswerable_count += 1

            refusal_pass = is_cautious_refusal(answer)
            refusal_passes += int(refusal_pass)

            print(f"Refusal pass: {refusal_pass}")

    print("\nRAG evaluation metrics")

    if answerable_count:
        print(
            "Retrieval accuracy: "
            f"{retrieval_passes / answerable_count:.3f}"
        )
        print(
            "Citation rate: "
            f"{citation_passes / answerable_count:.3f}"
        )
        print(
            "Expected-content rate: "
            f"{content_passes / answerable_count:.3f}"
        )

    if unanswerable_count:
        print(
            "Unsupported-question refusal rate: "
            f"{refusal_passes / unanswerable_count:.3f}"
        )


if __name__ == "__main__":
    main()