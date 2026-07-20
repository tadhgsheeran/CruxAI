from app.orchestration.router import Intent
from app.orchestration.semantic_router import (
    semantic_route_request,
)


def test_semantic_router_rejects_empty_request():
    decision = semantic_route_request("   ")

    assert decision.intent == Intent.UNSUPPORTED_REQUEST
    assert decision.tools == []


def test_semantic_router_returns_valid_decision():
    decision = semantic_route_request(
        "Find board climbs that resemble this setup."
    )

    assert isinstance(decision.intent, Intent)
    assert isinstance(decision.tools, list)
    assert decision.reason


def test_semantic_router_handles_training_paraphrase():
    decision = semantic_route_request(
        "Give me drills for improving on steep walls."
    )

    assert decision.intent in {
        Intent.TRAINING_RECOMMENDATION,
        Intent.GENERAL_CLIMBING_QUESTION,
    }
    
def test_semantic_router_routes_similar_board_request():
    decision = semantic_route_request(
        "Find board climbs that resemble this setup."
    )

    assert decision.intent == Intent.SIMILAR_ROUTE_SEARCH
    assert decision.tools == ["similar_route_search"]


def test_semantic_router_routes_multi_step_clauses():
    decision = semantic_route_request(
        "Predict this route grade and find similar routes."
    )

    assert decision.intent == Intent.MULTI_STEP_ANALYSIS
    assert decision.tools == [
        "grade_prediction",
        "similar_route_search",
    ]