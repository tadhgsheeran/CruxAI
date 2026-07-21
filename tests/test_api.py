import numpy as np
from fastapi.testclient import TestClient

from app.main import app

from app.generation.service import generation_service

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "CruxAI API is running."
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
    }


def test_predict_grade_with_real_route():
    route = np.load("data/example_test_route.npy")

    response = client.post(
        "/predict-grade",
        json={
            "holds": route.astype(int).tolist()
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert "predicted_grade" in result
    assert "rounded_grade" in result
    assert "formatted_grade" in result
    assert "model_version" in result

    assert abs(result["predicted_grade"] - 6.1741242) < 0.001
    assert result["rounded_grade"] == 6
    assert result["formatted_grade"] == "V6"
    assert result["model_version"] == "1.0.0"


def test_rejects_wrong_number_of_rows():
    invalid_route = [
        [0] * 11
        for _ in range(17)
    ]

    response = client.post(
        "/predict-grade",
        json={"holds": invalid_route},
    )

    assert response.status_code == 422


def test_rejects_wrong_number_of_columns():
    invalid_route = [
        [0] * 10
        for _ in range(18)
    ]

    response = client.post(
        "/predict-grade",
        json={"holds": invalid_route},
    )

    assert response.status_code == 422
    
# retrieve testing

def test_retrieve_returns_relevant_results():
    response = client.post(
        "/retrieve",
        json={
            "query": "How can I keep my feet on the wall on steep climbs?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["query"] == (
        "How can I keep my feet on the wall on steep climbs?"
    )
    assert len(result["results"]) == 3

    top_result = result["results"][0]

    assert top_result["source"] == "body_tension.md"
    assert "text" in top_result
    assert "chunk_id" in top_result
    assert "score" in top_result


def test_retrieve_respects_top_k():
    response = client.post(
        "/retrieve",
        json={
            "query": "How should I improve finger strength?",
            "top_k": 1,
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert len(result["results"]) == 1
    assert result["results"][0]["source"] == "finger_strength.md"


def test_retrieve_rejects_empty_query():
    response = client.post(
        "/retrieve",
        json={
            "query": "",
            "top_k": 3,
        },
    )

    assert response.status_code == 422


def test_retrieve_rejects_invalid_top_k():
    response = client.post(
        "/retrieve",
        json={
            "query": "How do I improve deadpoint timing?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422

# LLM test
    
def test_ask_returns_grounded_answer(monkeypatch):
    def fake_generate_answer(
        query: str,
        retrieved_results: list[dict],
        max_new_tokens: int = 250,
    ) -> str:
        return (
            "Your feet may cut loose because you are losing body tension. "
            "Focus on keeping pressure through your feet and core.\n\n"
            "Sources: [body_tension.md]"
        )

    monkeypatch.setattr(
        generation_service,
        "generate_answer",
        fake_generate_answer,
    )

    response = client.post(
        "/ask",
        json={
            "query": "Why do my feet cut loose on overhangs?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "Why do my feet cut loose on overhangs?"
    assert "body tension" in data["answer"].lower()
    assert "[body_tension.md]" in data["answer"]
    assert len(data["sources"]) == 3
    assert "body_tension.md" in data["sources"]

def test_ask_rejects_empty_query():
    response = client.post(
        "/ask",
        json={
            "query": "",
            "top_k": 3,
        },
    )

    assert response.status_code == 422


def test_ask_rejects_invalid_top_k():
    response = client.post(
        "/ask",
        json={
            "query": "How do I improve my footwork?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422
    
# unsupported quetsions test

def test_ask_refuses_unsupported_question():
    response = client.post(
        "/ask",
        json={
            "query": "What is the weather at Yosemite tomorrow?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "not have enough relevant information" in data["answer"].lower()
    assert data["sources"] == []
    
def test_route_endpoint_returns_decision(
    monkeypatch,
):
    from app.orchestration.router import (
        Intent,
        RoutingDecision,
    )

    def fake_semantic_route_request(question):
        return RoutingDecision(
            intent=Intent.TRAINING_RECOMMENDATION,
            tools=["training_recommendation"],
            reason="The request asks for training advice.",
        )

    monkeypatch.setattr(
        "app.main.semantic_route_request",
        fake_semantic_route_request,
    )

    response = client.post(
        "/route",
        json={
            "question": (
                "What drills should I use "
                "for steep climbing?"
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["intent"]
        == "TRAINING_RECOMMENDATION"
    )

    assert body["tools"] == [
        "training_recommendation"
    ]

    assert body["reason"]


def test_route_endpoint_rejects_empty_question():
    response = client.post(
        "/route",
        json={"question": ""},
    )

    assert response.status_code == 422
    
def test_analyze_endpoint_executes_workflow(
    monkeypatch,
):
    from app.orchestration.workflow_schemas import (
        WorkflowResponse,
    )
    from app.tools.schemas import ToolResult

    def fake_run_workflow(request):
        return WorkflowResponse(
            intent="MULTI_STEP_ANALYSIS",
            selected_tools=[
                "grade_prediction",
                "similar_route_search",
            ],
            success=True,
            tool_results={
                "grade_prediction": ToolResult(
                    tool="grade_prediction",
                    success=True,
                    summary="Predicted V6.",
                    data={
                        "formatted_grade": "V6",
                    },
                ),
                "similar_route_search": ToolResult(
                    tool="similar_route_search",
                    success=True,
                    summary="Found similar routes.",
                    data={
                        "matches": [],
                    },
                ),
            },
            final_answer=None,
            errors=[],
            metadata={
                "tools_attempted": 2,
                "tools_succeeded": 2,
            },
        )

    monkeypatch.setattr(
        "app.main.run_workflow",
        fake_run_workflow,
    )

    response = client.post(
        "/analyze",
        json={
            "question": (
                "Predict this route grade "
                "and find similar routes."
            ),
            "route": [
                [0] * 11
                for _ in range(18)
            ],
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["success"] is True
    assert body["intent"] == "MULTI_STEP_ANALYSIS"

    assert body["selected_tools"] == [
        "grade_prediction",
        "similar_route_search",
    ]

    assert (
        body["tool_results"]
        ["grade_prediction"]
        ["data"]
        ["formatted_grade"]
        == "V6"
    )

    assert body["metadata"]["tools_attempted"] == 2


def test_analyze_endpoint_accepts_training_context(
    monkeypatch,
):
    from app.orchestration.workflow_schemas import (
        WorkflowResponse,
    )

    captured_request = {}

    def fake_run_workflow(request):
        captured_request["request"] = request

        return WorkflowResponse(
            intent="TRAINING_RECOMMENDATION",
            selected_tools=[
                "training_recommendation"
            ],
            success=True,
            tool_results={},
            final_answer=None,
            errors=[],
            metadata={},
        )

    monkeypatch.setattr(
        "app.main.run_workflow",
        fake_run_workflow,
    )

    response = client.post(
        "/analyze",
        json={
            "question": (
                "What should I train "
                "to reach V7?"
            ),
            "current_grade": 5,
            "target_grade": 7,
            "top_k": 4,
        },
    )

    assert response.status_code == 200

    workflow_request = captured_request["request"]

    assert workflow_request.current_grade == 5
    assert workflow_request.target_grade == 7
    assert workflow_request.top_k == 4


def test_analyze_endpoint_rejects_empty_question():
    response = client.post(
        "/analyze",
        json={
            "question": "",
        },
    )

    assert response.status_code == 422


def test_analyze_endpoint_rejects_invalid_top_k():
    response = client.post(
        "/analyze",
        json={
            "question": "Explain body tension.",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_analyze_endpoint_rejects_negative_grade():
    response = client.post(
        "/analyze",
        json={
            "question": "What should I train?",
            "current_grade": -1,
        },
    )

    assert response.status_code == 422