import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.orchestration.router import Intent, RoutingDecision
from ingestion.embeddings import load_embedding_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEVELOPMENT_BENCHMARK_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "router_benchmark.jsonl"
)


SINGLE_INTENT_TO_TOOLS = {
    Intent.GENERAL_CLIMBING_QUESTION.value: [
        "knowledge_retrieval"
    ],
    Intent.ROUTE_GRADE_PREDICTION.value: [
        "grade_prediction"
    ],
    Intent.SIMILAR_ROUTE_SEARCH.value: [
        "similar_route_search"
    ],
    Intent.TRAINING_RECOMMENDATION.value: [
        "training_recommendation"
    ],
    Intent.UNSUPPORTED_REQUEST.value: [],
}


TOOL_TO_INTENT = {
    "grade_prediction": Intent.ROUTE_GRADE_PREDICTION,
    "similar_route_search": Intent.SIMILAR_ROUTE_SEARCH,
    "training_recommendation": Intent.TRAINING_RECOMMENDATION,
}


def load_single_intent_examples() -> list[dict]:
    """
    Load only examples representing one routing capability.

    Multi-step examples are excluded because their language overlaps
    several categories and can distort nearest-neighbor routing.
    """
    examples = []

    with DEVELOPMENT_BENCHMARK_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid routing example on line "
                    f"{line_number}: {exc}"
                ) from exc

            if (
                example["expected_intent"]
                != Intent.MULTI_STEP_ANALYSIS.value
            ):
                examples.append(example)

    if not examples:
        raise ValueError(
            "No single-intent routing examples were found."
        )

    return examples


@lru_cache(maxsize=1)
def build_semantic_index() -> dict:
    """
    Embed and cache the single-intent routing examples.
    """
    model = load_embedding_model()
    examples = load_single_intent_examples()

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


def split_request(question: str) -> list[str]:
    """
    Split a potentially multi-step request into smaller clauses.
    """
    clauses = re.split(
        r"\s*(?:,|\band\b|\bthen\b|\balso\b)\s*",
        question,
        flags=re.IGNORECASE,
    )

    return [
        clause.strip()
        for clause in clauses
        if clause.strip()
    ]


def classify_clause(
    clause: str,
    neighbors: int = 3,
) -> tuple[str, float]:
    """
    Classify one clause using weighted nearest-neighbor voting.
    """
    index = build_semantic_index()

    model = index["model"]
    examples = index["examples"]
    embeddings = index["embeddings"]

    query_embedding = model.encode(
        clause,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    similarities = embeddings @ query_embedding

    neighbor_count = min(
        neighbors,
        len(examples),
    )

    nearest_indices = np.argsort(
        similarities,
    )[::-1][:neighbor_count]

    intent_scores: dict[str, float] = {}

    for rank, example_index in enumerate(
        nearest_indices,
        start=1,
    ):
        example = examples[int(example_index)]
        intent_name = example["expected_intent"]

        weighted_score = (
            float(similarities[example_index])
            / rank
        )

        intent_scores[intent_name] = (
            intent_scores.get(intent_name, 0.0)
            + weighted_score
        )

    predicted_intent = max(
        intent_scores,
        key=intent_scores.get,
    )

    return (
        predicted_intent,
        intent_scores[predicted_intent],
    )


def semantic_route_request(
    question: str,
    neighbors: int = 3,
) -> RoutingDecision:
    """
    Route a request using clause-level semantic classification.
    """
    normalized_question = question.strip()

    if not normalized_question:
        return RoutingDecision(
            intent=Intent.UNSUPPORTED_REQUEST,
            tools=[],
            reason="The request is empty.",
        )

    clauses = split_request(normalized_question)

    clause_predictions = [
        classify_clause(
            clause=clause,
            neighbors=neighbors,
        )
        for clause in clauses
    ]

    predicted_intents = [
        prediction[0]
        for prediction in clause_predictions
    ]

    supported_tools = []

    for intent_name in predicted_intents:
        tools = SINGLE_INTENT_TO_TOOLS[intent_name]

        for tool in tools:
            if (
                tool != "knowledge_retrieval"
                and tool not in supported_tools
            ):
                supported_tools.append(tool)

    if len(supported_tools) >= 2:
        return RoutingDecision(
            intent=Intent.MULTI_STEP_ANALYSIS,
            tools=supported_tools,
            reason=(
                "Separate parts of the request require "
                "multiple CruxAI capabilities."
            ),
        )

    if len(supported_tools) == 1:
        selected_tool = supported_tools[0]

        return RoutingDecision(
            intent=TOOL_TO_INTENT[selected_tool],
            tools=[selected_tool],
            reason=(
                "The request is semantically closest to the "
                f"{selected_tool} capability."
            ),
        )

    if (
        Intent.GENERAL_CLIMBING_QUESTION.value
        in predicted_intents
    ):
        return RoutingDecision(
            intent=Intent.GENERAL_CLIMBING_QUESTION,
            tools=["knowledge_retrieval"],
            reason=(
                "The request is semantically closest to "
                "general climbing knowledge questions."
            ),
        )

    return RoutingDecision(
        intent=Intent.UNSUPPORTED_REQUEST,
        tools=[],
        reason=(
            "The request is semantically closest to "
            "unsupported examples."
        ),
    )