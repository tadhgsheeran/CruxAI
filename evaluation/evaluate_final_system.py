import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.retrieval.service import retrieval_service
from evaluation.evaluate_prompts import evaluate_prompt
from evaluation.evaluate_retrieval import evaluate_retrieval
from evaluation.evaluate_hybrid_router import (
    evaluate_hybrid_router,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "stage5_final_benchmark.json"
)

ROUTE_PATH_CANDIDATES = [
    PROJECT_ROOT / "example_test_route.npy",
    PROJECT_ROOT / "data" / "example_test_route.npy",
    PROJECT_ROOT / "tests" / "fixtures" / "example_test_route.npy",
]

FINAL_CONFIGURATION = {
    "retrieval_method": "hybrid",
    "hybrid_dense_weight": 0.90,
    "hybrid_keyword_weight": 0.10,
    "chunk_size": 500,
    "overlap_paragraphs": 2,
    "top_k": 3,
    "reranking_enabled": False,
    "prompt_version": "evidence_grounded",
    "generation_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "generation_dtype": "float32",
    "generation_device": "cpu",
    "max_new_tokens": 100,
}


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    """
    Calculate a percentile without requiring pandas.
    """
    if not values:
        return 0.0

    return float(
        np.percentile(
            values,
            percentile_value,
        )
    )


def find_example_route() -> list[list[int]] | None:
    """
    Find and load the example 18x11 route if available.
    """
    for path in ROUTE_PATH_CANDIDATES:
        if not path.exists():
            continue

        route = np.load(path)

        if route.shape != (18, 11):
            raise ValueError(
                f"Expected route shape (18, 11), "
                f"but {path} has shape {route.shape}."
            )

        return route.astype(int).tolist()

    return None


def summarize_latencies(
    latencies: list[float],
) -> dict[str, float]:
    """
    Produce common latency statistics.
    """
    if not latencies:
        return {
            "requests": 0,
            "mean_seconds": 0.0,
            "median_seconds": 0.0,
            "p50_seconds": 0.0,
            "p95_seconds": 0.0,
            "minimum_seconds": 0.0,
            "maximum_seconds": 0.0,
        }

    return {
        "requests": len(latencies),
        "mean_seconds": statistics.mean(latencies),
        "median_seconds": statistics.median(latencies),
        "p50_seconds": percentile(latencies, 50),
        "p95_seconds": percentile(latencies, 95),
        "minimum_seconds": min(latencies),
        "maximum_seconds": max(latencies),
    }


def build_api_cases() -> list[dict[str, Any]]:
    """
    Build representative API requests.
    """
    cases: list[dict[str, Any]] = [
        {
            "name": "knowledge_question",
            "payload": {
                "question": (
                    "Explain how body tension helps "
                    "on overhangs."
                ),
                "top_k": 3,
            },
        },
        {
            "name": "training_question",
            "payload": {
                "question": (
                    "What should I train to improve "
                    "on steep routes?"
                ),
                "top_k": 3,
                "current_grade": 5,
            },
        },
        {
            "name": "unsupported_question",
            "payload": {
                "question": (
                    "What is the weather at "
                    "Yosemite tomorrow?"
                ),
                "top_k": 3,
            },
        },
    ]

    route = find_example_route()

    if route is not None:
        cases.extend(
            [
                {
                    "name": "grade_and_difficulty",
                    "payload": {
                        "question": (
                            "How hard is this route and "
                            "what makes it difficult?"
                        ),
                        "route": route,
                        "top_k": 3,
                    },
                },
                {
                    "name": "full_multi_step",
                    "payload": {
                        "question": (
                            "How hard is this route, "
                            "what makes it difficult, "
                            "and what should I train?"
                        ),
                        "route": route,
                        "top_k": 3,
                        "current_grade": 5,
                    },
                },
            ]
        )

    return cases


def benchmark_analyze_endpoint(
    warmup_runs: int = 1,
    measured_runs: int = 3,
) -> dict:
    """
    Benchmark representative /analyze workflows.

    Each case gets one warmup by default, followed by
    three measured requests.
    """
    client = TestClient(app)
    cases = build_api_cases()

    case_results = []
    all_latencies = []
    successful_requests = 0
    total_requests = 0

    for case in cases:
        name = case["name"]
        payload = case["payload"]

        for _ in range(warmup_runs):
            response = client.post(
                "/analyze",
                json=payload,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Warmup failed for {name}: "
                    f"{response.status_code} "
                    f"{response.text}"
                )

        latencies = []
        status_codes = []
        workflow_latencies = []
        selected_tools = []
        intents = []

        for _ in range(measured_runs):
            start = time.perf_counter()

            response = client.post(
                "/analyze",
                json=payload,
            )

            elapsed = time.perf_counter() - start

            total_requests += 1
            status_codes.append(
                response.status_code
            )

            if response.status_code == 200:
                successful_requests += 1

            latencies.append(elapsed)
            all_latencies.append(elapsed)

            if response.status_code != 200:
                continue

            data = response.json()

            intent = data.get("intent")

            if intent:
                intents.append(intent)

            tools = (
                data.get("selected_tools")
                or data.get("tools")
                or []
            )

            if tools:
                selected_tools.append(tools)

            metadata = data.get("metadata", {})

            workflow_latency = (
                metadata.get("total_latency_seconds")
                or metadata.get("latency_seconds")
                or metadata.get("total_latency")
            )

            if isinstance(
                workflow_latency,
                (int, float),
            ):
                workflow_latencies.append(
                    float(workflow_latency)
                )

        case_result = {
            "name": name,
            "payload_summary": {
                "question": payload["question"],
                "has_route": "route" in payload,
                "top_k": payload.get("top_k"),
                "current_grade": payload.get(
                    "current_grade"
                ),
            },
            "status_codes": status_codes,
            "success_rate": (
                sum(
                    code == 200
                    for code in status_codes
                )
                / len(status_codes)
                if status_codes
                else 0.0
            ),
            "http_latency": summarize_latencies(
                latencies
            ),
            "reported_workflow_latency": (
                summarize_latencies(
                    workflow_latencies
                )
            ),
            "observed_intents": intents,
            "observed_tool_selections": (
                selected_tools
            ),
        }

        case_results.append(case_result)

        print(
            f"case={name:22} | "
            f"success="
            f"{case_result['success_rate']:.3f} | "
            f"p50="
            f"{case_result['http_latency']['p50_seconds']:.2f}s | "
            f"p95="
            f"{case_result['http_latency']['p95_seconds']:.2f}s"
        )

    return {
        "warmup_runs_per_case": warmup_runs,
        "measured_runs_per_case": measured_runs,
        "case_count": len(cases),
        "route_case_included": (
            find_example_route() is not None
        ),
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "success_rate": (
            successful_requests / total_requests
            if total_requests
            else 0.0
        ),
        "overall_http_latency": (
            summarize_latencies(
                all_latencies
            )
        ),
        "cases": case_results,
    }


def main() -> None:
    print("=" * 70)
    print("STAGE 5 FINAL BENCHMARK")
    print("=" * 70)

    print()
    print("Evaluating router...")
    router_metrics = evaluate_hybrid_router(
        verbose=False
    )

    print()
    print("Evaluating retrieval...")
    retrieval_metrics = evaluate_retrieval(
        top_k=3,
        verbose=False,
        service=retrieval_service,
        rerank=False,
        candidate_k=8,
    )

    print(
        f"retrieval | "
        f"hit_rate="
        f"{retrieval_metrics['hit_rate']:.3f} | "
        f"precision="
        f"{retrieval_metrics['precision_at_k']:.3f} | "
        f"recall="
        f"{retrieval_metrics['recall_at_k']:.3f} | "
        f"MRR="
        f"{retrieval_metrics['mean_reciprocal_rank']:.3f}"
    )

    print()
    print("Evaluating generation prompt...")
    generation_metrics = evaluate_prompt(
        "evidence_grounded"
    )

    print(
        f"generation | "
        f"topic_coverage="
        f"{generation_metrics['mean_topic_coverage']:.3f} | "
        f"citation_presence="
        f"{generation_metrics['citation_presence_rate']:.3f} | "
        f"unsupported_abstention="
        f"{generation_metrics['unsupported_abstention_rate']:.3f} | "
        f"false_refusal="
        f"{generation_metrics['supported_false_refusal_rate']:.3f} | "
        f"latency="
        f"{generation_metrics['average_latency_seconds']:.2f}s"
    )

    print()
    print("Benchmarking end-to-end workflows...")
    end_to_end_metrics = (
        benchmark_analyze_endpoint(
            warmup_runs=1,
            measured_runs=3,
        )
    )

    output = {
        "experiment": "stage5_final_benchmark",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "configuration": FINAL_CONFIGURATION,
        "router_metrics": router_metrics,
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "end_to_end_metrics": (
            end_to_end_metrics
        ),
        "selected_decisions": {
            "retrieval": (
                "Use 90/10 dense-heavy hybrid retrieval."
            ),
            "chunking": (
                "Use 500-character chunks with "
                "two-paragraph overlap."
            ),
            "top_k": (
                "Return three retrieved sources."
            ),
            "reranking": (
                "Disable cross-encoder reranking because "
                "it reduced retrieval coverage and greatly "
                "increased latency."
            ),
            "prompt": (
                "Use the evidence-grounded prompt, with "
                "router-based rejection and automatic "
                "source attachment enforced outside the LLM."
            ),
            "quantization": (
                "Use FP32 on the tested Apple CPU because "
                "INT8 weight-only quantization reduced "
                "storage but substantially worsened latency "
                "and topic coverage."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    overall_latency = end_to_end_metrics[
        "overall_http_latency"
    ]

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Router intent accuracy: "
        f"{router_metrics.get('intent_accuracy', 0.0):.3f}"
    )

    print(
        f"Router tool accuracy: "
        f"{router_metrics.get('tool_selection_accuracy', 0.0):.3f}"
    )

    print(
        f"Retrieval Precision@3: "
        f"{retrieval_metrics['precision_at_k']:.3f}"
    )

    print(
        f"Retrieval Recall@3: "
        f"{retrieval_metrics['recall_at_k']:.3f}"
    )

    print(
        f"Retrieval MRR: "
        f"{retrieval_metrics['mean_reciprocal_rank']:.3f}"
    )

    print(
        f"End-to-end success rate: "
        f"{end_to_end_metrics['success_rate']:.3f}"
    )

    print(
        f"End-to-end P50 latency: "
        f"{overall_latency['p50_seconds']:.2f}s"
    )

    print(
        f"End-to-end P95 latency: "
        f"{overall_latency['p95_seconds']:.2f}s"
    )

    print()
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()