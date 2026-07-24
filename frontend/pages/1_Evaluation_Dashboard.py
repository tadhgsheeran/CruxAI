import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

FINAL_BENCHMARK_PATH = (
    RESULTS_DIR
    / "stage5_final_benchmark.json"
)

st.set_page_config(
    page_title="CruxAI Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("CruxAI Evaluation Dashboard")

st.write(
    "Interactive summary of the Stage 5 optimization "
    "and benchmarking experiments."
)


def load_json(path: Path) -> dict[str, Any] | None:
    """
    Load a JSON file safely.
    """
    if not path.exists():
        return None

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ) as exc:
        st.warning(
            f"Could not load {path.name}: {exc}"
        )

        return None


def get_results_list(
    data: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    Extract a list of experiment rows from common
    Stage 5 result-file structures.
    """
    if not data:
        return []

    results = data.get("results")

    if isinstance(results, list):
        return results

    metrics = data.get("metrics")

    if isinstance(metrics, list):
        return metrics

    return []


def display_dataframe(
    rows: list[dict[str, Any]],
    columns: list[str],
    labels: dict[str, str],
) -> None:
    """
    Display selected fields from experiment rows.
    """
    if not rows:
        st.info(
            "No results were found for this experiment."
        )
        return

    filtered_rows = []

    for row in rows:
        filtered_rows.append(
            {
                labels.get(column, column): row.get(column)
                for column in columns
                if column in row
            }
        )

    dataframe = pd.DataFrame(filtered_rows)

    if dataframe.empty:
        st.info(
            "The results file did not contain "
            "the expected fields."
        )
        return

    st.markdown(
        dataframe.to_html(
            index=False,
            border=0,
        ),
        unsafe_allow_html=True,
    )


def display_metric_chart(
    rows: list[dict[str, Any]],
    index_column: str,
    metric_columns: list[str],
) -> None:
    """
    Plot selected experiment metrics.
    """
    if not rows:
        return

    records = []

    for row in rows:
        record = {
            index_column: row.get(index_column)
        }

        for metric in metric_columns:
            if metric in row:
                record[metric] = row[metric]

        records.append(record)

    dataframe = pd.DataFrame(records)

    if (
        dataframe.empty
        or index_column not in dataframe.columns
    ):
        return

    available_metrics = [
        metric
        for metric in metric_columns
        if metric in dataframe.columns
    ]

    if not available_metrics:
        return

    dataframe = dataframe.set_index(
        index_column
    )

    st.line_chart(
        dataframe[available_metrics]
    )


final_benchmark = load_json(
    FINAL_BENCHMARK_PATH
)

st.header("Final System Benchmark")

if final_benchmark:
    metrics = final_benchmark.get(
        "metrics",
        final_benchmark,
    )

    router = metrics.get(
        "router",
        {},
    )

    retrieval = metrics.get(
        "retrieval",
        {},
    )

    latency = metrics.get(
        "latency",
        metrics.get(
            "evaluation_latency",
            {},
        ),
    )

    metric_columns = st.columns(6)

    router_accuracy = router.get(
        "intent_accuracy",
        0,
    )

    tool_accuracy = router.get(
        "tool_selection_accuracy",
        0,
    )

    hit_rate = retrieval.get(
        "hit_rate",
        0,
    )

    precision_at_k = retrieval.get(
        "precision_at_k",
        0,
    )

    recall_at_k = retrieval.get(
        "recall_at_k",
        0,
    )

    mrr = retrieval.get(
        "mean_reciprocal_rank",
        0,
    )

    metric_columns[0].metric(
        "Router accuracy",
        f"{router_accuracy:.3f}",
    )

    metric_columns[1].metric(
        "Tool accuracy",
        f"{tool_accuracy:.3f}",
    )

    metric_columns[2].metric(
        "Hit rate",
        f"{hit_rate:.3f}",
    )

    metric_columns[3].metric(
        "Precision@k",
        f"{precision_at_k:.3f}",
    )

    metric_columns[4].metric(
        "Recall@k",
        f"{recall_at_k:.3f}",
    )

    metric_columns[5].metric(
        "MRR",
        f"{mrr:.3f}",
    )

    with st.expander(
        "View final benchmark JSON"
    ):
        st.json(final_benchmark)

else:
    st.warning(
        "stage5_final_benchmark.json was not found."
    )


st.divider()

st.header("Top-k Ablation")

top_k_data = load_json(
    RESULTS_DIR
    / "top_k_ablation.json"
)

top_k_results = get_results_list(
    top_k_data
)

display_dataframe(
    rows=top_k_results,
    columns=[
        "top_k",
        "hit_rate",
        "precision_at_k",
        "recall_at_k",
        "mean_reciprocal_rank",
        "failure_count",
    ],
    labels={
        "top_k": "Top-k",
        "hit_rate": "Hit rate",
        "precision_at_k": "Precision@k",
        "recall_at_k": "Recall@k",
        "mean_reciprocal_rank": "MRR",
        "failure_count": "Failures",
    },
)

display_metric_chart(
    rows=top_k_results,
    index_column="top_k",
    metric_columns=[
        "hit_rate",
        "precision_at_k",
        "recall_at_k",
        "mean_reciprocal_rank",
    ],
)


st.divider()

st.header("Chunk-size Ablation")

chunk_data = load_json(
    RESULTS_DIR
    / "chunk_size_ablation.json"
)

chunk_results = get_results_list(
    chunk_data
)

display_dataframe(
    rows=chunk_results,
    columns=[
        "chunk_size",
        "chunk_count",
        "hit_rate",
        "precision_at_k",
        "recall_at_k",
        "mean_reciprocal_rank",
        "index_build_seconds",
    ],
    labels={
        "chunk_size": "Chunk size",
        "chunk_count": "Chunk count",
        "hit_rate": "Hit rate",
        "precision_at_k": "Precision@k",
        "recall_at_k": "Recall@k",
        "mean_reciprocal_rank": "MRR",
        "index_build_seconds": "Build seconds",
    },
)

display_metric_chart(
    rows=chunk_results,
    index_column="chunk_size",
    metric_columns=[
        "hit_rate",
        "precision_at_k",
        "recall_at_k",
        "mean_reciprocal_rank",
    ],
)


st.divider()

st.header("Overlap Ablation")

overlap_data = load_json(
    RESULTS_DIR
    / "overlap_ablation.json"
)

overlap_results = get_results_list(
    overlap_data
)

display_dataframe(
    rows=overlap_results,
    columns=[
        "overlap_paragraphs",
        "chunk_count",
        "hit_rate",
        "precision_at_k",
        "recall_at_k",
        "mean_reciprocal_rank",
    ],
    labels={
        "overlap_paragraphs": "Overlap",
        "chunk_count": "Chunk count",
        "hit_rate": "Hit rate",
        "precision_at_k": "Precision@k",
        "recall_at_k": "Recall@k",
        "mean_reciprocal_rank": "MRR",
    },
)

display_metric_chart(
    rows=overlap_results,
    index_column="overlap_paragraphs",
    metric_columns=[
        "hit_rate",
        "precision_at_k",
        "recall_at_k",
        "mean_reciprocal_rank",
    ],
)


st.divider()

st.header("Retrieval-method Comparison")

retrieval_method_data = load_json(
    RESULTS_DIR
    / "retrieval_method_ablation.json"
)

retrieval_method_results = get_results_list(
    retrieval_method_data
)

display_dataframe(
    rows=retrieval_method_results,
    columns=[
        "retrieval_method",
        "method",
        "hit_rate",
        "precision_at_k",
        "recall_at_k",
        "mean_reciprocal_rank",
        "evaluation_seconds",
    ],
    labels={
        "retrieval_method": "Method",
        "method": "Method",
        "hit_rate": "Hit rate",
        "precision_at_k": "Precision@k",
        "recall_at_k": "Recall@k",
        "mean_reciprocal_rank": "MRR",
        "evaluation_seconds": "Evaluation seconds",
    },
)

method_index = (
    "retrieval_method"
    if any(
        "retrieval_method" in result
        for result in retrieval_method_results
    )
    else "method"
)

display_metric_chart(
    rows=retrieval_method_results,
    index_column=method_index,
    metric_columns=[
        "hit_rate",
        "precision_at_k",
        "recall_at_k",
        "mean_reciprocal_rank",
    ],
)


st.divider()

st.header("Hybrid-weight Ablation")

hybrid_data = load_json(
    RESULTS_DIR
    / "hybrid_weight_ablation.json"
)

hybrid_results = get_results_list(
    hybrid_data
)

display_dataframe(
    rows=hybrid_results,
    columns=[
        "hybrid_dense_weight",
        "dense_weight",
        "keyword_weight",
        "hit_rate",
        "precision_at_k",
        "recall_at_k",
        "mean_reciprocal_rank",
    ],
    labels={
        "hybrid_dense_weight": "Dense weight",
        "dense_weight": "Dense weight",
        "keyword_weight": "Keyword weight",
        "hit_rate": "Hit rate",
        "precision_at_k": "Precision@k",
        "recall_at_k": "Recall@k",
        "mean_reciprocal_rank": "MRR",
    },
)

hybrid_index = (
    "hybrid_dense_weight"
    if any(
        "hybrid_dense_weight" in result
        for result in hybrid_results
    )
    else "dense_weight"
)

display_metric_chart(
    rows=hybrid_results,
    index_column=hybrid_index,
    metric_columns=[
        "hit_rate",
        "precision_at_k",
        "recall_at_k",
        "mean_reciprocal_rank",
    ],
)


st.divider()

st.header("Reranking Ablation")

reranking_data = load_json(
    RESULTS_DIR
    / "reranking_ablation.json"
)

reranking_results = get_results_list(
    reranking_data
)

display_dataframe(
    rows=reranking_results,
    columns=[
        "reranking_enabled",
        "configuration",
        "hit_rate",
        "precision_at_k",
        "recall_at_k",
        "mean_reciprocal_rank",
        "evaluation_seconds",
    ],
    labels={
        "reranking_enabled": "Reranking",
        "configuration": "Configuration",
        "hit_rate": "Hit rate",
        "precision_at_k": "Precision@k",
        "recall_at_k": "Recall@k",
        "mean_reciprocal_rank": "MRR",
        "evaluation_seconds": "Evaluation seconds",
    },
)


st.divider()

st.header("Prompt Ablation")

prompt_data = load_json(
    RESULTS_DIR
    / "prompt_ablation.json"
)

prompt_results = get_results_list(
    prompt_data
)

display_dataframe(
    rows=prompt_results,
    columns=[
        "prompt_version",
        "valid_response_rate",
        "citation_rate",
        "citation_correctness",
        "topic_coverage",
        "average_latency_seconds",
    ],
    labels={
        "prompt_version": "Prompt",
        "valid_response_rate": "Valid response rate",
        "citation_rate": "Citation rate",
        "citation_correctness": "Citation correctness",
        "topic_coverage": "Topic coverage",
        "average_latency_seconds": "Average latency",
    },
)


st.divider()

st.header("Quantization Ablation")

quantization_data = load_json(
    RESULTS_DIR
    / "quantization_ablation.json"
)

quantization_results = get_results_list(
    quantization_data
)

display_dataframe(
    rows=quantization_results,
    columns=[
        "quantization",
        "mode",
        "success",
        "model_load_seconds",
        "generation_seconds",
        "total_seconds",
        "tokens_per_second",
        "memory_mb",
        "error",
    ],
    labels={
        "quantization": "Quantization",
        "mode": "Mode",
        "success": "Success",
        "model_load_seconds": "Load seconds",
        "generation_seconds": "Generation seconds",
        "total_seconds": "Total seconds",
        "tokens_per_second": "Tokens/second",
        "memory_mb": "Memory MB",
        "error": "Error",
    },
)


st.divider()

st.header("Selected Deployment Configuration")

configuration_columns = st.columns(2)

with configuration_columns[0]:
    st.subheader("Retrieval")

    st.markdown(
        """
- **Method:** Dense retrieval
- **Chunk size:** 500 characters
- **Overlap:** 2 paragraphs
- **Top-k:** 3
- **Reranking:** Disabled
        """
    )

with configuration_columns[1]:
    st.subheader("Generation and Serving")

    st.markdown(
        """
- **Local generation:** Qwen2.5-0.5B-Instruct
- **Public CPU mode:** Deterministic evidence synthesis
- **API:** FastAPI
- **Frontend:** Streamlit
- **Deployment:** Docker
        """
    )

st.info(
    "The CPU deployment uses deterministic evidence synthesis "
    "because containerized Qwen inference required roughly seven "
    "minutes per response. The deterministic deployment completed "
    "the same workflow in approximately two seconds."
)