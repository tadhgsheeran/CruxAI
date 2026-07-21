import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.benchmark_config import (
    BASELINE_CONFIG,
    BenchmarkConfig,
)

from evaluation.evaluate_retrieval import (
    evaluate_retrieval,
)
from evaluation.evaluate_router import (
    evaluate_router,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    """
    Calculate a percentile using linear interpolation.
    """
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (
        percentile_value / 100
    ) * (len(ordered) - 1)

    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(ordered) - 1,
    )

    fraction = position - lower_index

    return (
        ordered[lower_index]
        + (
            ordered[upper_index]
            - ordered[lower_index]
        )
        * fraction
    )


def summarize_latencies(
    latencies: list[float],
) -> dict[str, float]:
    """
    Produce common latency statistics.
    """
    if not latencies:
        return {
            "count": 0,
            "mean_seconds": 0.0,
            "median_seconds": 0.0,
            "p50_seconds": 0.0,
            "p95_seconds": 0.0,
            "minimum_seconds": 0.0,
            "maximum_seconds": 0.0,
        }

    return {
        "count": len(latencies),
        "mean_seconds": statistics.mean(latencies),
        "median_seconds": statistics.median(latencies),
        "p50_seconds": percentile(latencies, 50),
        "p95_seconds": percentile(latencies, 95),
        "minimum_seconds": min(latencies),
        "maximum_seconds": max(latencies),
    }


def build_result_record(
    config: BenchmarkConfig,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Combine experiment configuration, environment details,
    and measured metrics.
    """
    return {
        "experiment": config.to_dict(),
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "metrics": metrics,
    }


def save_result(
    result: dict[str, Any],
    experiment_name: str,
) -> Path:
    """
    Save one benchmark result as JSON.
    """
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_path = (
        RESULTS_DIR
        / f"{experiment_name}_{timestamp}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    return output_path


def main() -> None:
    """
    Run and save the CruxAI Stage 5 baseline benchmark.
    """
    config = BASELINE_CONFIG

    benchmark_start = time.perf_counter()

    router_start = time.perf_counter()

    router_metrics = evaluate_router(
        verbose=False,
    )

    router_evaluation_seconds = (
        time.perf_counter() - router_start
    )

    retrieval_start = time.perf_counter()

    retrieval_metrics = evaluate_retrieval(
        top_k=config.top_k,
        verbose=False,
    )

    retrieval_evaluation_seconds = (
        time.perf_counter() - retrieval_start
    )

    total_benchmark_seconds = (
        time.perf_counter() - benchmark_start
    )

    metrics = {
        "router": router_metrics,
        "retrieval": retrieval_metrics,
        "evaluation_latency": {
            "router_seconds": (
                router_evaluation_seconds
            ),
            "retrieval_seconds": (
                retrieval_evaluation_seconds
            ),
            "total_seconds": (
                total_benchmark_seconds
            ),
        },
    }

    result = build_result_record(
        config=config,
        metrics=metrics,
    )

    output_path = save_result(
        result=result,
        experiment_name=config.experiment_name,
    )

    print("=" * 70)
    print("CRUXAI STAGE 5 BASELINE")
    print("=" * 70)

    print("\nRouter")
    print(
        "Examples:",
        router_metrics["examples"],
    )
    print(
        "Intent accuracy:",
        f"{router_metrics['intent_accuracy']:.3f}",
    )
    print(
        "Tool-selection accuracy:",
        f"{router_metrics['tool_selection_accuracy']:.3f}",
    )

    print("\nRetrieval")
    print(
        "Questions:",
        retrieval_metrics["questions"],
    )
    print(
        "Top-k:",
        retrieval_metrics["top_k"],
    )
    print(
        "Hit rate:",
        f"{retrieval_metrics['hit_rate']:.3f}",
    )
    print(
        "Precision@k:",
        f"{retrieval_metrics['precision_at_k']:.3f}",
    )
    print(
        "Recall@k:",
        f"{retrieval_metrics['recall_at_k']:.3f}",
    )
    print(
        "Mean reciprocal rank:",
        f"{retrieval_metrics['mean_reciprocal_rank']:.3f}",
    )

    print("\nEvaluation timing")
    print(
        "Router:",
        f"{router_evaluation_seconds:.3f} seconds",
    )
    print(
        "Retrieval:",
        f"{retrieval_evaluation_seconds:.3f} seconds",
    )
    print(
        "Total:",
        f"{total_benchmark_seconds:.3f} seconds",
    )

    print()
    print(f"Saved result to: {output_path}")

if __name__ == "__main__":
    main()