import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.orchestration.router import Intent, RoutingDecision
from ingestion.embeddings import load_embedding_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BENCHMARK_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "router_benchmark.jsonl"
)


TOOL_INTENTS = {
    "grade_prediction": Intent.ROUTE_GRADE_PREDICTION,
    "similar_route_search": Intent.SIMILAR_ROUTE_SEARCH,
    "training_recommendation": Intent.TRAINING_RECOMMENDATION,
}


@lru_cache(maxsize=1)
def build_hybrid_index() -> dict:
    model = load_embedding_model()

    examples = []

    with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                examples.append(json.loads(line))

    questions = [
        example["question"]
        for example in examples
    ]

    embeddings = model.encode(
        questions,
        normalize_embeddings=True,
    )

    return {
        "model": model,
        "examples": examples,
        "embeddings": np.asarray(
            embeddings,
            dtype=np.float32,
        ),
    }


def calculate_tool_scores(
    question: str,
) -> tuple[dict[str, float], float, float]:
    index = build_hybrid_index()

    model = index["model"]
    examples = index["examples"]
    embeddings = index["embeddings"]

    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    similarities = embeddings @ np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    tool_scores = {
        "grade_prediction": -1.0,
        "similar_route_search": -1.0,
        "training_recommendation": -1.0,
    }

    general_score = -1.0
    unsupported_score = -1.0

    for example, similarity in zip(
        examples,
        similarities,
    ):
        similarity = float(similarity)
        intent = example["expected_intent"]

        if intent == Intent.GENERAL_CLIMBING_QUESTION.value:
            general_score = max(
                general_score,
                similarity,
            )

        if intent == Intent.UNSUPPORTED_REQUEST.value:
            unsupported_score = max(
                unsupported_score,
                similarity,
            )

        for tool in example["expected_tools"]:
            if tool in tool_scores:
                tool_scores[tool] = max(
                    tool_scores[tool],
                    similarity,
                )

    return tool_scores, general_score, unsupported_score


def hybrid_route_request(
    question: str,
    tool_threshold: float = 0.52,
    margin: float = 0.03,
) -> RoutingDecision:
    normalized_question = question.strip()

    if not normalized_question:
        return RoutingDecision(
            intent=Intent.UNSUPPORTED_REQUEST,
            tools=[],
            reason="The request is empty.",
        )

    tool_scores, general_score, unsupported_score = (
        calculate_tool_scores(normalized_question)
    )

    selected_tools = [
        tool
        for tool, score in tool_scores.items()
        if score >= tool_threshold
    ]

    # Require supported climbing evidence to beat unsupported
    # evidence by at least a small margin.
    best_supported_score = max(
        [general_score, *tool_scores.values()]
    )

    if (
        unsupported_score
        > best_supported_score + margin
    ):
        return RoutingDecision(
            intent=Intent.UNSUPPORTED_REQUEST,
            tools=[],
            reason=(
                "The request is semantically closer to "
                "unsupported examples than climbing examples."
            ),
        )

    if len(selected_tools) >= 2:
        return RoutingDecision(
            intent=Intent.MULTI_STEP_ANALYSIS,
            tools=selected_tools,
            reason=(
                "The request contains multiple independently "
                "detected CruxAI capabilities."
            ),
        )

    if len(selected_tools) == 1:
        selected_tool = selected_tools[0]

        return RoutingDecision(
            intent=TOOL_INTENTS[selected_tool],
            tools=[selected_tool],
            reason=(
                f"The request most strongly matches the "
                f"{selected_tool} capability."
            ),
        )

    if general_score >= tool_threshold:
        return RoutingDecision(
            intent=Intent.GENERAL_CLIMBING_QUESTION,
            tools=["knowledge_retrieval"],
            reason=(
                "The request is most similar to general "
                "climbing knowledge questions."
            ),
        )

    return RoutingDecision(
        intent=Intent.UNSUPPORTED_REQUEST,
        tools=[],
        reason=(
            "The router did not find enough evidence that "
            "the request belongs to a supported category."
        ),
    )