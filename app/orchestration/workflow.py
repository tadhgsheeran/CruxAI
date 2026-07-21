from time import perf_counter

from app.orchestration.semantic_router import (
    semantic_route_request,
)
from app.orchestration.tool_executor import (
    execute_tool,
)
from app.orchestration.workflow_schemas import (
    WorkflowRequest,
    WorkflowResponse,
    WorkflowState,
)

from app.orchestration.response_synthesizer import (
    synthesize_response,
)

from app.tools.factor_training import (
    build_factor_training_recommendations,
)

def run_workflow(
    request: WorkflowRequest,
) -> WorkflowResponse:
    """
    Route a request, execute the selected tools, and return
    the complete workflow state.
    """
    workflow_start = perf_counter()

    state = WorkflowState(
        request=request,
    )

    routing_start = perf_counter()

    decision = semantic_route_request(
        request.question,
    )

    routing_latency = perf_counter() - routing_start

    state.intent = decision.intent.value
    state.selected_tools = decision.tools
    
    route_analysis_tools = {
        "grade_prediction",
        "similar_route_search",
    }

    if (
        request.route is not None
        and route_analysis_tools.intersection(
            state.selected_tools
        )
        and "difficulty_analysis"
        not in state.selected_tools
    ):
        state.selected_tools.insert(
            1,
            "difficulty_analysis",
        )

    if not state.selected_tools:
        state.errors.append(
            decision.reason
        )

        total_latency = perf_counter() - workflow_start

        return WorkflowResponse(
            intent=state.intent,
            selected_tools=[],
            success=False,
            tool_results={},
            final_answer=None,
            errors=state.errors,
            metadata={
                "routing_reason": decision.reason,
                "routing_latency_seconds": routing_latency,
                "total_latency_seconds": total_latency,
                "tools_attempted": 0,
                "tools_succeeded": 0,
            },
        )

    tool_latencies = {}

    execution_order = [
        tool_name
        for tool_name in [
            "grade_prediction",
            "difficulty_analysis",
            "similar_route_search",
            "knowledge_retrieval",
            "training_recommendation",
        ]
        if tool_name in state.selected_tools
    ]

    for tool_name in execution_order:
        tool_start = perf_counter()

        tool_request = request

        if tool_name == "training_recommendation":
            request_updates = {}

            if request.target_grade is None:
                grade_result = state.tool_results.get(
                    "grade_prediction"
                )

                if (
                    grade_result is not None
                    and grade_result.success
                ):
                    predicted_target_grade = (
                        grade_result.data.get(
                            "rounded_grade"
                        )
                    )

                    if predicted_target_grade is not None:
                        request_updates[
                            "target_grade"
                        ] = int(
                            predicted_target_grade
                        )

            difficulty_result = state.tool_results.get(
                "difficulty_analysis"
            )

            if (
                difficulty_result is not None
                and difficulty_result.success
            ):
                difficulty_factors = (
                    difficulty_result.data.get(
                        "difficulty_factors",
                        [],
                    )
                )

                factor_training = (
                    build_factor_training_recommendations(
                        difficulty_result.data
                    )
                )
    
                if difficulty_factors:
                    request_updates[
                        "difficulty_factors"
                    ] = difficulty_factors

                if factor_training:
                    request_updates[
                        "factor_training_recommendations"
                    ] = factor_training

            if request_updates:
                tool_request = request.model_copy(
                    update=request_updates
                )
        
        try:
            result = execute_tool(
                tool_name=tool_name,
                request=tool_request,
            )

        except Exception as exc:
            state.errors.append(
                f"{tool_name}: {exc}"
            )

            tool_latencies[tool_name] = (
                perf_counter() - tool_start
            )
    
            continue

        tool_latencies[tool_name] = (
            perf_counter() - tool_start
        )

        state.tool_results[tool_name] = result

        if not result.success:
            state.errors.append(
                f"{tool_name}: "
                f"{result.error or result.summary}"
            )

    successful_results = [
        result
        for result in state.tool_results.values()
        if result.success
    ]

    failed_results = [
        result
        for result in state.tool_results.values()
        if not result.success
    ]

    state.final_answer = synthesize_response(
        question=request.question,
        tool_results=state.tool_results,
    )
    
    total_latency = perf_counter() - workflow_start

    workflow_success = (
        len(successful_results) > 0
        and len(failed_results) == 0
        and len(state.errors) == 0
    )
    
    return WorkflowResponse(
        intent=state.intent,
        selected_tools=state.selected_tools,
        success=workflow_success,
        tool_results=state.tool_results,
        final_answer=state.final_answer,
        errors=state.errors,
        metadata={
            "routing_reason": decision.reason,
            "routing_latency_seconds": routing_latency,
            "tool_latencies_seconds": tool_latencies,
            "total_latency_seconds": total_latency,
            "tools_attempted": len(
                state.selected_tools
            ),
            "tools_succeeded": len(
                successful_results
            ),
            "tools_failed": (
                len(state.selected_tools)
                - len(successful_results)
            ),
        },
    )