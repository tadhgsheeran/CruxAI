import re
from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    GENERAL_CLIMBING_QUESTION = "GENERAL_CLIMBING_QUESTION"
    ROUTE_GRADE_PREDICTION = "ROUTE_GRADE_PREDICTION"
    SIMILAR_ROUTE_SEARCH = "SIMILAR_ROUTE_SEARCH"
    TRAINING_RECOMMENDATION = "TRAINING_RECOMMENDATION"
    MULTI_STEP_ANALYSIS = "MULTI_STEP_ANALYSIS"
    UNSUPPORTED_REQUEST = "UNSUPPORTED_REQUEST"


class RoutingDecision(BaseModel):
    intent: Intent
    tools: list[str] = Field(default_factory=list)
    reason: str


GRADE_KEYWORDS = {
    "grade",
    "difficulty",
    "difficult",
    "how hard",
    "hard is",
    "v grade",
    "predict",
    "rate this route",
    "rate this climb",
    "estimate",
    "realistic",
}

SIMILAR_ROUTE_KEYWORDS = {
    "similar route",
    "similar routes",
    "similar climb",
    "similar climbs",
    "similar holds",
    "similar hold",
    "similar layout",
    "similar hold layout",
    "routes like",
    "climbs like",
    "problems like",
    "find routes",
    "find climbs",
    "find problems",
    "compare routes",
    "compare this route",
    "same holds",
    "closely match",
    "closest match",
    "match this climb",
}

TRAINING_KEYWORDS = {
    "train",
    "training",
    "workout",
    "exercise",
    "exercises",
    "improve",
    "weakness",
    "weaknesses",
    "practice",
    "session",
    "get stronger",
    "prepare",
}

CLIMBING_TOPIC_KEYWORDS = {
    "climb",
    "climbing",
    "climber",
    "route",
    "problem",
    "moonboard",
    "hold",
    "holds",
    "heel hook",
    "deadpoint",
    "lock off",
    "lock-off",
    "body tension",
    "footwork",
    "sloper",
    "crimp",
    "overhang",
    "steep",
    "steep terrain",
    "power endurance",
    "finger strength",
    "hip",
    "hips",
    "hip position",
    "hip positioning",
}

UNSUPPORTED_KEYWORDS = {
    "weather",
    "stock",
    "stocks",
    "recipe",
    "politics",
    "political",
    "news",
    "movie",
    "football",
    "basketball",
}


def contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def contains_v_grade(text: str) -> bool:
    """
    Detect grades written as V0, V5, V10, and similar forms.
    """
    return re.search(r"\bv\d+\b", text) is not None


def route_request(question: str) -> RoutingDecision:
    """
    Classify a user request and select the appropriate CruxAI tool.
    """
    normalized_question = question.strip().lower()

    if not normalized_question:
        return RoutingDecision(
            intent=Intent.UNSUPPORTED_REQUEST,
            tools=[],
            reason="The request is empty.",
        )

    if contains_any(
        normalized_question,
        UNSUPPORTED_KEYWORDS,
    ):
        return RoutingDecision(
            intent=Intent.UNSUPPORTED_REQUEST,
            tools=[],
            reason=(
                "The request is outside CruxAI's climbing "
                "and route-analysis capabilities."
            ),
        )

    has_climbing_context = contains_any(
        normalized_question,
        CLIMBING_TOPIC_KEYWORDS,
    )

    has_similar_route_intent = contains_any(
        normalized_question,
        SIMILAR_ROUTE_KEYWORDS,
    )

    has_training_intent = contains_any(
        normalized_question,
        TRAINING_KEYWORDS,
    )
    
    has_explicit_grade_request = contains_any(
        normalized_question,
        GRADE_KEYWORDS,
    )

    has_route_reference = contains_any(
        normalized_question,
        {
            "route",
            "climb",
            "problem",
            "holds",
            "hold layout",
        },
    )

    has_grade_intent = (
        has_explicit_grade_request
        or (
            contains_v_grade(normalized_question)
            and has_route_reference
            and not has_training_intent
        )
    )

    detected_intents = sum(
        [
            has_grade_intent,
            has_similar_route_intent,
            has_training_intent,
        ]
    )

    if detected_intents >= 2:
        tools = []

        if has_grade_intent:
            tools.append("grade_prediction")

        if has_similar_route_intent:
            tools.append("similar_route_search")

        if has_training_intent:
            tools.append("training_recommendation")

        return RoutingDecision(
            intent=Intent.MULTI_STEP_ANALYSIS,
            tools=tools,
            reason=(
                "The request requires more than one CruxAI "
                "capability."
            ),
        )

    if has_similar_route_intent:
        return RoutingDecision(
            intent=Intent.SIMILAR_ROUTE_SEARCH,
            tools=["similar_route_search"],
            reason=(
                "The request asks to find or compare "
                "similar climbing routes."
            ),
        )

    if has_grade_intent:
        return RoutingDecision(
            intent=Intent.ROUTE_GRADE_PREDICTION,
            tools=["grade_prediction"],
            reason=(
                "The request asks for a route difficulty "
                "or grade prediction."
            ),
        )

    if has_training_intent:
        return RoutingDecision(
            intent=Intent.TRAINING_RECOMMENDATION,
            tools=["training_recommendation"],
            reason=(
                "The request asks for training advice "
                "or improvement recommendations."
            ),
        )

    if has_climbing_context:
        return RoutingDecision(
            intent=Intent.GENERAL_CLIMBING_QUESTION,
            tools=["knowledge_retrieval"],
            reason=(
                "The request asks for general climbing "
                "knowledge or technique information."
            ),
        )

    return RoutingDecision(
        intent=Intent.UNSUPPORTED_REQUEST,
        tools=[],
        reason=(
            "The request does not appear related to climbing "
            "or route analysis."
        ),
    )