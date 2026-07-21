import json
import re
import time
from pathlib import Path

from app.generation.service import (
    generation_service,
)
from app.retrieval.service import (
    retrieval_service,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "prompt_benchmark.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "prompt_ablation.json"
)

ABSTENTION_TEXT = (
    "not have enough relevant information"
)


def load_benchmark() -> list[dict]:
    examples = []

    with BENCHMARK_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                examples.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc

    return examples


def extract_citations(
    answer: str,
) -> list[str]:
    """
    Extract source filenames written in square brackets.
    """
    return re.findall(
        r"\[([A-Za-z0-9_.-]+\.md)\]",
        answer,
    )


def calculate_topic_coverage(
    answer: str,
    expected_topics: list[str],
) -> float:
    if not expected_topics:
        return 1.0

    normalized_answer = answer.lower()

    covered = sum(
        topic.lower() in normalized_answer
        for topic in expected_topics
    )

    return covered / len(expected_topics)


def evaluate_prompt(
    prompt_version: str,
) -> dict:
    examples = load_benchmark()

    supported_example_count = 0
    unsupported_example_count = 0

    unsupported_abstention_scores = []
    supported_false_refusal_scores = []
    
    citation_present_scores = []
    citation_validity_scores = []
    expected_source_citation_scores = []
    topic_coverage_scores = []
    abstention_scores = []
    response_lengths = []
    latencies = []

    example_results = []

    for example in examples:
        question = example["question"]
        expected_sources = set(
            example["relevant_sources"]
        )
        expected_topics = example[
            "expected_topics"
        ]
        should_abstain = example[
            "should_abstain"
        ]

        if should_abstain:
            retrieved_results = []
        else:
            retrieved_results = (
                retrieval_service.search(
                    query=question,
                    top_k=3,
                )
            )

        start = time.perf_counter()

        answer = generation_service.generate_answer(
            query=question,
            retrieved_results=retrieved_results,
            max_new_tokens=100,
            prompt_version=prompt_version,
            append_missing_citations=False,
        )

        latency = time.perf_counter() - start

        citations = extract_citations(answer)
        citation_set = set(citations)

        retrieved_source_set = {
            result["source"]
            for result in retrieved_results
        }
        
        abstained = (
            ABSTENTION_TEXT
            in answer.lower()
        )

        if should_abstain:
            unsupported_example_count += 1

            abstention_correct = abstained
            topic_coverage = 1.0

            unsupported_abstention_scores.append(
                float(abstained)
            )

            citation_present = None
            citation_valid = None
            expected_source_cited = None

        else:
            supported_example_count += 1

            abstention_correct = not abstained

            supported_false_refusal_scores.append(
                float(abstained)
            )

            citation_present = bool(citations)

            citation_valid = (
                citation_set.issubset(
                    retrieved_source_set
                )
                if citations
                else False
            )

            expected_source_cited = bool(
                citation_set & expected_sources
            )

            topic_coverage = (
                calculate_topic_coverage(
                    answer=answer,
                    expected_topics=expected_topics,
                )
            )

            citation_present_scores.append(
                float(citation_present)
            )

            citation_validity_scores.append(
                float(citation_valid)
            )

            expected_source_citation_scores.append(
                float(expected_source_cited)
            )

        topic_coverage_scores.append(
            topic_coverage
        )

        abstention_scores.append(
            float(abstention_correct)
        )

        response_lengths.append(
            len(answer.split())
        )

        latencies.append(latency)

        example_results.append(
            {
                "question": question,
                "should_abstain": (
                    should_abstain
                ),
                "abstained": abstained,
                "expected_sources": sorted(
                    expected_sources
                ),
                "citations": citations,
                "citation_present": (
                    citation_present
                ),
                "citation_valid": citation_valid,
                "expected_source_cited": expected_source_cited,
                
                "topic_coverage": (
                    topic_coverage
                ),
                "response_words": len(
                    answer.split()
                ),
                "latency_seconds": latency,
                "answer": answer,
            }
        )

    total = len(examples)

    return {
        "prompt_version": prompt_version,
        "examples": total,
        "supported_examples": supported_example_count,
        "unsupported_examples": unsupported_example_count,

        "citation_presence_rate": (
            sum(citation_present_scores)
            / supported_example_count
            if supported_example_count
            else 0.0
        ),

        "citation_validity_rate": (
            sum(citation_validity_scores)
            / supported_example_count
            if supported_example_count
            else 0.0
        ),

        "expected_source_citation_rate": (
            sum(expected_source_citation_scores)
            / supported_example_count
            if supported_example_count
            else 0.0
        ),

        "unsupported_abstention_rate": (
            sum(unsupported_abstention_scores)
            / unsupported_example_count
            if unsupported_example_count
            else 0.0
        ),

        "supported_false_refusal_rate": (
            sum(supported_false_refusal_scores)
            / supported_example_count
            if supported_example_count
            else 0.0
        ),
        
        "mean_topic_coverage": (
            sum(topic_coverage_scores)
            / total
            if total
            else 0.0
        ),
        "abstention_accuracy": (
            sum(abstention_scores)
            / total
            if total
            else 0.0
        ),
        "average_response_words": (
            sum(response_lengths)
            / total
            if total
            else 0.0
        ),
        "average_latency_seconds": (
            sum(latencies)
            / total
            if total
            else 0.0
        ),
        "examples_detail": example_results,
    }


def main() -> None:
    prompt_versions = [
        "baseline",
        "evidence_grounded",
    ]

    results = []

    for prompt_version in prompt_versions:
        metrics = evaluate_prompt(
            prompt_version
        )

        results.append(metrics)

        print(
            f"prompt={prompt_version:19} | "
            f"citations="
            f"{metrics['citation_presence_rate']:.3f} | "
            f"valid_citations="
            f"{metrics['citation_validity_rate']:.3f} | "
            f"expected_source="
            f"{metrics['expected_source_citation_rate']:.3f} | "
            f"topic_coverage="
            f"{metrics['mean_topic_coverage']:.3f} | "
            f"unsupported_abstention="
            f"{metrics['unsupported_abstention_rate']:.3f} | "
            f"false_refusal="
            f"{metrics['supported_false_refusal_rate']:.3f} | "
            f"words="
            f"{metrics['average_response_words']:.1f} | "
            f"latency="
            f"{metrics['average_latency_seconds']:.2f}s"
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
                "experiment": "prompt_ablation",
                "model": (
                    "Qwen/Qwen2.5-0.5B-Instruct"
                ),
                "max_new_tokens": 100,
                "retrieval_method": "hybrid",
                "results": results,
            },
            file,
            indent=2,
        )

    print()
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()