from app.orchestration.router import (
    Intent,
    RoutingDecision,
)
from app.orchestration.workflow import (
    run_workflow,
)
from app.orchestration.workflow_schemas import (
    WorkflowRequest,
)
from app.tools.schemas import ToolResult


def test_workflow_executes_single_tool(
    monkeypatch,
):
    def fake_router(question):
        return RoutingDecision(
            intent=Intent.GENERAL_CLIMBING_QUESTION,
            tools=["knowledge_retrieval"],
            reason="General climbing question.",
        )

    def fake_execute_tool(tool_name, request):
        return ToolResult(
            tool="knowledge_retrieval",
            success=True,
            summary="Retrieved relevant knowledge.",
            data={
                "query": request.question,
            },
        )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "semantic_route_request",
        fake_router,
    )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "execute_tool",
        fake_execute_tool,
    )

    request = WorkflowRequest(
        question="Explain body tension.",
    )

    response = run_workflow(request)

    assert response.success is True
    assert (
        response.intent
        == "GENERAL_CLIMBING_QUESTION"
    )
    assert response.selected_tools == [
        "knowledge_retrieval"
    ]

    assert "knowledge_retrieval" in (
        response.tool_results
    )

    assert response.errors == []
    assert response.metadata["tools_attempted"] == 1
    assert response.metadata["tools_succeeded"] == 1


def test_workflow_executes_multiple_tools(
    monkeypatch,
):
    def fake_router(question):
        return RoutingDecision(
            intent=Intent.MULTI_STEP_ANALYSIS,
            tools=[
                "grade_prediction",
                "similar_route_search",
            ],
            reason="Multiple capabilities required.",
        )

    def fake_execute_tool(tool_name, request):
        if tool_name == "grade_prediction":
            return ToolResult(
                tool="grade_prediction",
                success=True,
                summary="Predicted V6.",
                data={
                    "formatted_grade": "V6",
                },
            )

        return ToolResult(
            tool="similar_route_search",
            success=True,
            summary="Found similar routes.",
            data={
                "matches": [],
            },
        )

    def fake_execute_tool(tool_name, request):
        if tool_name == "grade_prediction":
            return ToolResult(
                tool="grade_prediction",
                success=True,
                summary="Predicted V6.",
                data={
                    "formatted_grade": "V6",
                },
            )

        if tool_name == "difficulty_analysis":
            return ToolResult(
                tool="difficulty_analysis",
                success=True,
                summary="Analyzed difficulty factors.",
                data={
                    "difficulty_factors": [
                        "The route contains a long move."
                    ],
                },
            )

        return ToolResult(
            tool="similar_route_search",
            success=True,
            summary="Found similar routes.",
            data={
            "matches": [],
            },
        )
    
    monkeypatch.setattr(
        "app.orchestration.workflow."
        "semantic_route_request",
        fake_router,
    )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "execute_tool",
        fake_execute_tool,
    )

    request = WorkflowRequest(
        question=(
            "Predict this route grade "
            "and find similar routes."
        ),
        route=[[0] * 11 for _ in range(18)],
    )

    response = run_workflow(request)

    assert response.success is True

    assert response.intent == "MULTI_STEP_ANALYSIS"

    assert response.selected_tools == [
        "grade_prediction",
        "difficulty_analysis",
        "similar_route_search",
    ]

    assert set(response.tool_results) == {
        "grade_prediction",
        "difficulty_analysis",
        "similar_route_search",
    }

    assert response.metadata["tools_attempted"] == 3
    assert response.metadata["tools_succeeded"] == 3


def test_workflow_handles_tool_failure(
    monkeypatch,
):
    def fake_router(question):
        return RoutingDecision(
            intent=Intent.MULTI_STEP_ANALYSIS,
            tools=[
                "grade_prediction",
                "similar_route_search",
            ],
            reason="Multiple capabilities required.",
        )

    def fake_execute_tool(tool_name, request):
        if tool_name == "grade_prediction":
            return ToolResult(
                tool="grade_prediction",
                success=True,
                summary="Predicted V7.",
            )
    
        if tool_name == "difficulty_analysis":
            return ToolResult(
                tool="difficulty_analysis",
                success=True,
                summary="Analyzed difficulty factors.",
                data={
                    "difficulty_factors": [
                    "The route has a wide span."
                    ],
                },
            )

        return ToolResult(
            tool="similar_route_search",
            success=False,
            summary="Search failed.",
            error="Route database unavailable.",
    )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "semantic_route_request",
        fake_router,
    )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "execute_tool",
        fake_execute_tool,
    )

    request = WorkflowRequest(
        question="Analyze this route.",
        route=[[0] * 11 for _ in range(18)],
    )

    response = run_workflow(request)

    assert response.success is False

    assert (
        response.tool_results[
            "grade_prediction"
        ].success
        is True
    )

    assert (
        response.tool_results[
            "similar_route_search"
        ].success
        is False
    )

    assert response.errors
    assert "Route database unavailable" in (
        response.errors[0]
    )

    assert response.metadata["tools_attempted"] == 3
    assert response.metadata["tools_succeeded"] == 2
    assert response.metadata["tools_failed"] == 1


def test_workflow_handles_executor_exception(
    monkeypatch,
):
    def fake_router(question):
        return RoutingDecision(
            intent=Intent.ROUTE_GRADE_PREDICTION,
            tools=["grade_prediction"],
            reason="Grade prediction requested.",
        )

    def fake_execute_tool(tool_name, request):
        raise RuntimeError(
            "Unexpected model failure."
        )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "semantic_route_request",
        fake_router,
    )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "execute_tool",
        fake_execute_tool,
    )

    request = WorkflowRequest(
        question="Predict this route.",
        route=[[0] * 11 for _ in range(18)],
    )

    response = run_workflow(request)

    assert response.success is False
    assert response.tool_results == {}
    assert response.errors
    assert "Unexpected model failure" in (
        response.errors[0]
    )

    assert response.metadata["tools_attempted"] == 2
    assert response.metadata["tools_succeeded"] == 0
    assert response.metadata["tools_failed"] == 2
    
    assert len(response.errors) == 2


def test_workflow_handles_unsupported_request(
    monkeypatch,
):
    def fake_router(question):
        return RoutingDecision(
            intent=Intent.UNSUPPORTED_REQUEST,
            tools=[],
            reason=(
                "The request is outside CruxAI's scope."
            ),
        )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "semantic_route_request",
        fake_router,
    )

    request = WorkflowRequest(
        question="Help me choose a laptop.",
    )

    response = run_workflow(request)

    assert response.success is False
    assert response.intent == "UNSUPPORTED_REQUEST"
    assert response.selected_tools == []
    assert response.tool_results == {}
    assert response.errors
    assert response.metadata["tools_attempted"] == 0


def test_workflow_metadata_contains_latency(
    monkeypatch,
):
    def fake_router(question):
        return RoutingDecision(
            intent=Intent.GENERAL_CLIMBING_QUESTION,
            tools=["knowledge_retrieval"],
            reason="General climbing question.",
        )

    def fake_execute_tool(tool_name, request):
        return ToolResult(
            tool="knowledge_retrieval",
            success=True,
            summary="Retrieved information.",
        )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "semantic_route_request",
        fake_router,
    )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "execute_tool",
        fake_execute_tool,
    )

    response = run_workflow(
        WorkflowRequest(
            question="What is a deadpoint?",
        )
    )

    assert (
        response.metadata[
            "routing_latency_seconds"
        ]
        >= 0
    )

    assert (
        response.metadata[
            "total_latency_seconds"
        ]
        >= 0
    )

    assert "knowledge_retrieval" in (
        response.metadata[
            "tool_latencies_seconds"
        ]
    )
    
def test_workflow_populates_final_answer(
    monkeypatch,
):
    def fake_router(question):
        return RoutingDecision(
            intent=Intent.ROUTE_GRADE_PREDICTION,
            tools=["grade_prediction"],
            reason="Grade prediction requested.",
        )

    def fake_execute_tool(tool_name, request):
        return ToolResult(
            tool="grade_prediction",
            success=True,
            summary="Predicted V6.",
            data={
                "predicted_grade": 6.2,
                "formatted_grade": "V6",
            },
        )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "semantic_route_request",
        fake_router,
    )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "execute_tool",
        fake_execute_tool,
    )

    response = run_workflow(
        WorkflowRequest(
            question="How hard is this route?",
            route=[[0] * 11 for _ in range(18)],
        )
    )

    assert response.final_answer is not None
    assert "V6" in response.final_answer
    
def test_training_uses_predicted_grade_as_target(
    monkeypatch,
):
    captured_requests = {}

    def fake_router(question):
        return RoutingDecision(
            intent=Intent.MULTI_STEP_ANALYSIS,
            tools=[
                "grade_prediction",
                "training_recommendation",
            ],
            reason="Grade prediction and training required.",
        )

    def fake_execute_tool(tool_name, request):
        captured_requests[tool_name] = request

        if tool_name == "grade_prediction":
            return ToolResult(
                tool="grade_prediction",
                success=True,
                summary="Predicted V7.",
                data={
                    "predicted_grade": 6.8,
                    "rounded_grade": 7,
                    "formatted_grade": "V7",
                },
            )

        return ToolResult(
            tool="training_recommendation",
            success=True,
            summary="Generated training advice.",
            data={
                "current_grade": request.current_grade,
                "target_grade": request.target_grade,
                "recommendation": (
                    "Train body tension and "
                    "power endurance."
                ),
            },
        )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "semantic_route_request",
        fake_router,
    )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "execute_tool",
        fake_execute_tool,
    )

    response = run_workflow(
        WorkflowRequest(
            question=(
                "How hard is this route "
                "and what should I train?"
            ),
            route=[
                [0] * 11
                for _ in range(18)
            ],
            current_grade=5,
        )
    )

    assert response.success is True

    training_request = captured_requests[
        "training_recommendation"
    ]

    assert training_request.current_grade == 5
    assert training_request.target_grade == 7

    training_result = response.tool_results[
        "training_recommendation"
    ]

    assert training_result.data[
        "target_grade"
    ] == 7
    
def test_explicit_target_grade_overrides_prediction(
    monkeypatch,
):
    captured_training_request = {}

    def fake_router(question):
        return RoutingDecision(
            intent=Intent.MULTI_STEP_ANALYSIS,
            tools=[
                "grade_prediction",
                "training_recommendation",
            ],
            reason="Multiple tools required.",
        )

    def fake_execute_tool(tool_name, request):
        if tool_name == "grade_prediction":
            return ToolResult(
                tool="grade_prediction",
                success=True,
                summary="Predicted V7.",
                data={
                    "rounded_grade": 7,
                    "formatted_grade": "V7",
                },
            )

        captured_training_request["request"] = request

        return ToolResult(
            tool="training_recommendation",
            success=True,
            summary="Generated training advice.",
            data={
                "target_grade": request.target_grade,
                "recommendation": "Training advice.",
            },
        )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "semantic_route_request",
        fake_router,
    )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "execute_tool",
        fake_execute_tool,
    )

    run_workflow(
        WorkflowRequest(
            question=(
                "Analyze this route and "
                "help me train for V8."
            ),
            route=[
                [0] * 11
                for _ in range(18)
            ],
            current_grade=5,
            target_grade=8,
        )
    )

    assert (
        captured_training_request[
            "request"
        ].target_grade
        == 8
    )
    
def test_workflow_adds_difficulty_analysis_for_route(
    monkeypatch,
):
    executed_tools = []

    def fake_router(question):
        return RoutingDecision(
            intent=Intent.ROUTE_GRADE_PREDICTION,
            tools=["grade_prediction"],
            reason="Grade prediction requested.",
        )

    def fake_execute_tool(tool_name, request):
        executed_tools.append(tool_name)

        if tool_name == "grade_prediction":
            return ToolResult(
                tool="grade_prediction",
                success=True,
                summary="Predicted V6.",
                data={
                    "rounded_grade": 6,
                    "formatted_grade": "V6",
                },
            )

        return ToolResult(
            tool="difficulty_analysis",
            success=True,
            summary="Analyzed difficulty factors.",
            data={
                "difficulty_factors": [
                    "The route has a wide span."
                ],
            },
        )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "semantic_route_request",
        fake_router,
    )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "execute_tool",
        fake_execute_tool,
    )

    response = run_workflow(
        WorkflowRequest(
            question="How hard is this route?",
            route=[
                [0] * 11
                for _ in range(18)
            ],
        )
    )

    assert response.success is True

    assert response.selected_tools == [
        "grade_prediction",
        "difficulty_analysis",
    ]

    assert executed_tools == [
        "grade_prediction",
        "difficulty_analysis",
    ]
    
def test_training_uses_difficulty_factors(
    monkeypatch,
):
    captured_training_request = {}

    def fake_router(question):
        return RoutingDecision(
            intent=Intent.MULTI_STEP_ANALYSIS,
            tools=[
                "grade_prediction",
                "training_recommendation",
            ],
            reason="Route analysis and training required.",
        )

    def fake_execute_tool(tool_name, request):
        if tool_name == "grade_prediction":
            return ToolResult(
                tool="grade_prediction",
                success=True,
                summary="Predicted V6.",
                data={
                    "rounded_grade": 6,
                    "formatted_grade": "V6",
                },
            )

        if tool_name == "difficulty_analysis":
            return ToolResult(
                tool="difficulty_analysis",
                success=True,
                summary="Analyzed route geometry.",
                data={
                    "difficulty_factors": [
                        (
                            "The estimated average move "
                            "distance is large."
                        ),
                        (
                            "The route contains at least one "
                            "especially long estimated move."
                        ),
                    ],
                },
            )

        captured_training_request[
            "request"
        ] = request

        return ToolResult(
            tool="training_recommendation",
            success=True,
            summary="Generated training advice.",
            data={
                "difficulty_factors": (
                    request.difficulty_factors
                ),
                "recommendation": (
                    "Train explosive pulling and "
                    "lock-off strength."
                ),
            },
        )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "semantic_route_request",
        fake_router,
    )

    monkeypatch.setattr(
        "app.orchestration.workflow."
        "execute_tool",
        fake_execute_tool,
    )

    response = run_workflow(
        WorkflowRequest(
            question=(
                "How hard is this route, "
                "what makes it difficult, "
                "and what should I train?"
            ),
            route=[
                [0] * 11
                for _ in range(18)
            ],
            current_grade=5,
        )
    )

    training_request = captured_training_request[
        "request"
    ]

    assert training_request.target_grade == 6

    assert len(
        training_request.difficulty_factors
    ) == 2

    assert "average move distance" in (
        training_request.difficulty_factors[0]
    )

    training_result = response.tool_results[
        "training_recommendation"
    ]

    assert len(
        training_result.data[
            "difficulty_factors"
        ]
    ) == 2