from app.orchestration.router import (
    Intent,
    route_request,
)


def test_routes_grade_prediction():
    decision = route_request(
        "Can you predict the grade of this route?"
    )

    assert decision.intent == Intent.ROUTE_GRADE_PREDICTION
    assert decision.tools == ["grade_prediction"]


def test_routes_similar_route_search():
    decision = route_request(
        "Find routes similar to this one."
    )

    assert decision.intent == Intent.SIMILAR_ROUTE_SEARCH
    assert decision.tools == ["similar_route_search"]


def test_routes_training_recommendation():
    decision = route_request(
        "What should I train to improve on overhangs?"
    )

    assert decision.intent == Intent.TRAINING_RECOMMENDATION
    assert decision.tools == ["training_recommendation"]


def test_routes_general_climbing_question():
    decision = route_request(
        "Explain how a heel hook works."
    )

    assert decision.intent == Intent.GENERAL_CLIMBING_QUESTION
    assert decision.tools == ["knowledge_retrieval"]


def test_routes_multi_step_analysis():
    decision = route_request(
        "How difficult is this route and what should I train?"
    )

    assert decision.intent == Intent.MULTI_STEP_ANALYSIS
    assert "grade_prediction" in decision.tools
    assert "training_recommendation" in decision.tools


def test_routes_unsupported_request():
    decision = route_request(
        "What is the weather today?"
    )

    assert decision.intent == Intent.UNSUPPORTED_REQUEST
    assert decision.tools == []


def test_routes_empty_request_as_unsupported():
    decision = route_request("   ")

    assert decision.intent == Intent.UNSUPPORTED_REQUEST
    assert decision.tools == []