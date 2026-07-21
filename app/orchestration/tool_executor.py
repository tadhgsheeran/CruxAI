from app.orchestration.workflow_schemas import (
    WorkflowRequest,
)
from app.tools.grade_prediction import (
    predict_grade_tool,
)
from app.tools.knowledge_retrieval import (
    knowledge_retrieval_tool,
)
from app.tools.similar_routes import (
    similar_route_search_tool,
)
from app.tools.training_recommendation import (
    training_recommendation_tool,
)
from app.tools.schemas import ToolResult

from app.tools.difficulty_analysis import (
    difficulty_analysis_tool,
)

SUPPORTED_TOOLS = {
    "grade_prediction",
    "difficulty_analysis",
    "knowledge_retrieval",
    "similar_route_search",
    "training_recommendation",
}


def missing_input_result(
    tool_name: str,
    message: str,
) -> ToolResult:
    """
    Return a standardized failure when required workflow
    input is missing.
    """
    return ToolResult(
        tool=tool_name,
        success=False,
        summary=f"{tool_name} could not be executed.",
        error=message,
    )

def execute_tool(
    tool_name: str,
    request: WorkflowRequest,
) -> ToolResult:
    """
    Execute one CruxAI tool using the workflow request.
    """
    if tool_name not in SUPPORTED_TOOLS:
        raise ValueError(
            f"Unsupported tool: {tool_name}"
        )

    if tool_name == "grade_prediction":
        if request.route is None:
            return missing_input_result(
                tool_name="grade_prediction",
                message=(
                    "A route is required for grade prediction."
                ),
            )

        return predict_grade_tool(
            route=request.route,
        )

    if tool_name == "difficulty_analysis":
        if request.route is None:
            return missing_input_result(
                tool_name="difficulty_analysis",
                message=(
                    "A route is required for difficulty analysis."
                ),
            )

        return difficulty_analysis_tool(
            route=request.route,
        )

    if tool_name == "similar_route_search":
        if request.route is None:
            return missing_input_result(
                tool_name="similar_route_search",
                message=(
                    "A route is required for similar-route "
                    "search."
                ),
            )

        return similar_route_search_tool(
            route=request.route,
            top_k=request.top_k,
        )

    if tool_name == "knowledge_retrieval":
        return knowledge_retrieval_tool(
            query=request.question,
            top_k=request.top_k,
        )

    if tool_name == "training_recommendation":
        return training_recommendation_tool(
            question=request.question,
            current_grade=request.current_grade,
            target_grade=request.target_grade,
            difficulty_factors=(
                request.difficulty_factors
            ),
            factor_training_recommendations=(
                request.factor_training_recommendations
            ),
            top_k=request.top_k,
        )

    raise RuntimeError(
        f"Tool execution was not implemented: {tool_name}"
    )