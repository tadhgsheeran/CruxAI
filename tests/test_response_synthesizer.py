from app.orchestration.response_synthesizer import (
    synthesize_response,
)
from app.tools.schemas import ToolResult, ToolSource


def test_synthesizes_grade_prediction():
    results = {
        "grade_prediction": ToolResult(
            tool="grade_prediction",
            success=True,
            summary="Predicted V6.",
            data={
                "predicted_grade": 6.24,
                "formatted_grade": "V6",
            },
        )
    }

    answer = synthesize_response(
        question="How hard is this route?",
        tool_results=results,
    )

    assert "V6" in answer
    assert "6.24" in answer


def test_synthesizes_similar_routes():
    results = {
        "similar_route_search": ToolResult(
            tool="similar_route_search",
            success=True,
            summary="Found similar routes.",
            data={
                "matches": [
                    {
                        "route_id": 42,
                        "formatted_grade": "V7",
                        "similarity": 0.75,
                    },
                    {
                        "route_id": 91,
                        "formatted_grade": "V6",
                        "similarity": 0.68,
                    },
                ]
            },
        )
    }

    answer = synthesize_response(
        question="Find similar routes.",
        tool_results=results,
    )

    assert "Route 42" in answer
    assert "V7" in answer
    assert "0.750" in answer


def test_synthesizes_training_recommendation():
    results = {
        "training_recommendation": ToolResult(
            tool="training_recommendation",
            success=True,
            summary="Generated advice.",
            data={
                "recommendation": (
                    "Train body tension and "
                    "power endurance."
                )
            },
        )
    }

    answer = synthesize_response(
        question="What should I train?",
        tool_results=results,
    )

    assert "body tension" in answer
    assert "power endurance" in answer


def test_synthesizes_retrieval_sources():
    results = {
        "knowledge_retrieval": ToolResult(
            tool="knowledge_retrieval",
            success=True,
            summary="Retrieved evidence.",
            sources=[
                ToolSource(
                    document="body_tension.md",
                    chunk_id="body_tension_0",
                    score=0.91,
                    text=(
                        "Body tension helps keep the "
                        "feet connected to the wall."
                    ),
                )
            ],
        )
    }

    answer = synthesize_response(
        question="Explain body tension.",
        tool_results=results,
    )

    assert "feet connected" in answer
    assert "[body_tension.md]" in answer


def test_synthesizes_multiple_tool_results():
    results = {
        "grade_prediction": ToolResult(
            tool="grade_prediction",
            success=True,
            summary="Predicted V7.",
            data={
                "predicted_grade": 7.1,
                "formatted_grade": "V7",
            },
        ),
        "training_recommendation": ToolResult(
            tool="training_recommendation",
            success=True,
            summary="Generated advice.",
            data={
                "recommendation": (
                    "Focus on lock-off strength."
                )
            },
        ),
    }

    answer = synthesize_response(
        question=(
            "How hard is this route and "
            "what should I train?"
        ),
        tool_results=results,
    )

    assert "V7" in answer
    assert "lock-off strength" in answer


def test_handles_no_successful_results():
    results = {
        "grade_prediction": ToolResult(
            tool="grade_prediction",
            success=False,
            summary="Prediction failed.",
            error="Route missing.",
        )
    }

    answer = synthesize_response(
        question="Predict this route.",
        tool_results=results,
    )

    assert "could not complete" in answer.lower()
    
def test_synthesizes_difficulty_analysis():
    results = {
        "difficulty_analysis": ToolResult(
            tool="difficulty_analysis",
            success=True,
            summary="Analyzed route difficulty.",
            data={
                "hold_count": 6,
                "vertical_span": 15,
                "horizontal_span": 8,
                "average_move_distance": 3.75,
                "difficulty_factors": [
                    (
                        "The route contains at least one "
                        "especially long estimated move."
                    ),
                    (
                        "The route has a wide horizontal span."
                    ),
                ],
            },
        )
    }

    answer = synthesize_response(
        question="What makes this route difficult?",
        tool_results=results,
    )

    assert "Route difficulty factors" in answer
    assert "Active holds: 6" in answer
    assert "Vertical span: 15" in answer
    assert "Horizontal span: 8" in answer
    assert "3.75" in answer
    assert "especially long" in answer