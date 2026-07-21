import numpy as np
import pytest

from app.orchestration.tool_executor import (
    execute_tool,
)
from app.orchestration.workflow_schemas import (
    WorkflowRequest,
)
from app.tools.schemas import ToolResult


def make_test_route() -> list[list[int]]:
    route = np.zeros(
        (18, 11),
        dtype=int,
    )

    route[1, 2] = 1
    route[4, 4] = 1
    route[7, 6] = 1
    route[10, 5] = 1
    route[13, 7] = 1
    route[16, 8] = 1

    return route.tolist()


def test_execute_grade_prediction_tool(
    monkeypatch,
):
    def fake_predict_grade_tool(route):
        return ToolResult(
            tool="grade_prediction",
            success=True,
            summary="Predicted V6.",
            data={
                "formatted_grade": "V6",
            },
        )

    monkeypatch.setattr(
        "app.orchestration.tool_executor."
        "predict_grade_tool",
        fake_predict_grade_tool,
    )

    request = WorkflowRequest(
        question="Predict this route's grade.",
        route=make_test_route(),
    )

    result = execute_tool(
        tool_name="grade_prediction",
        request=request,
    )

    assert result.success is True
    assert result.tool == "grade_prediction"
    assert result.data["formatted_grade"] == "V6"


def test_grade_prediction_requires_route():
    request = WorkflowRequest(
        question="Predict this route's grade.",
    )

    result = execute_tool(
        tool_name="grade_prediction",
        request=request,
    )

    assert result.success is False
    assert result.tool == "grade_prediction"
    assert "route is required" in result.error.lower()


def test_similar_route_search_requires_route():
    request = WorkflowRequest(
        question="Find similar routes.",
    )

    result = execute_tool(
        tool_name="similar_route_search",
        request=request,
    )

    assert result.success is False
    assert result.tool == "similar_route_search"
    assert "route is required" in result.error.lower()


def test_execute_knowledge_retrieval(
    monkeypatch,
):
    def fake_retrieval_tool(query, top_k):
        return ToolResult(
            tool="knowledge_retrieval",
            success=True,
            summary="Retrieved climbing information.",
            data={
                "query": query,
                "top_k": top_k,
            },
        )

    monkeypatch.setattr(
        "app.orchestration.tool_executor."
        "knowledge_retrieval_tool",
        fake_retrieval_tool,
    )

    request = WorkflowRequest(
        question="Explain body tension.",
        top_k=4,
    )

    result = execute_tool(
        tool_name="knowledge_retrieval",
        request=request,
    )

    assert result.success is True
    assert result.data["top_k"] == 4


def test_execute_training_recommendation(
    monkeypatch,
):
    def fake_training_tool(
        question,
        current_grade,
        target_grade,
        difficulty_factors,
        factor_training_recommendations,
        top_k,
    ):
        return ToolResult(
            tool="training_recommendation",
            success=True,
            summary="Generated training advice.",
            data={
                "current_grade": current_grade,
                "target_grade": target_grade,
                "difficulty_factors": difficulty_factors,
                "factor_training_recommendations": (
                    factor_training_recommendations
                ),
            },
        )

    monkeypatch.setattr(
        "app.orchestration.tool_executor."
        "training_recommendation_tool",
        fake_training_tool,
    )

    request = WorkflowRequest(
        question="What should I train?",
        current_grade=5,
        target_grade=7,
    )

    result = execute_tool(
        tool_name="training_recommendation",
        request=request,
    )

    assert result.success is True
    assert result.data["current_grade"] == 5
    assert result.data["target_grade"] == 7


def test_execute_tool_rejects_unknown_tool():
    request = WorkflowRequest(
        question="Test question.",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported tool",
    ):
        execute_tool(
            tool_name="unknown_tool",
            request=request,
        )
        
def test_execute_difficulty_analysis(
    monkeypatch,
):
    def fake_difficulty_analysis_tool(route):
        return ToolResult(
            tool="difficulty_analysis",
            success=True,
            summary="Analyzed difficulty factors.",
            data={
                "hold_count": 6,
                "difficulty_factors": [
                    "The route contains a long move."
                ],
            },
        )

    monkeypatch.setattr(
        "app.orchestration.tool_executor."
        "difficulty_analysis_tool",
        fake_difficulty_analysis_tool,
    )

    request = WorkflowRequest(
        question="What makes this route difficult?",
        route=make_test_route(),
    )

    result = execute_tool(
        tool_name="difficulty_analysis",
        request=request,
    )

    assert result.success is True
    assert result.tool == "difficulty_analysis"
    assert result.data["hold_count"] == 6


def test_difficulty_analysis_requires_route():
    request = WorkflowRequest(
        question="What makes this route difficult?",
    )

    result = execute_tool(
        tool_name="difficulty_analysis",
        request=request,
    )

    assert result.success is False
    assert "route is required" in result.error.lower()