from typing import Any

from pydantic import BaseModel, Field

from app.tools.schemas import ToolResult

class WorkflowRequest(BaseModel):
    """
    Information that may be supplied to a CruxAI workflow.
    """

    question: str

    route: list[list[int | float]] | None = None

    current_grade: int | None = None
    target_grade: int | None = None

    difficulty_factors: list[str] = Field(
        default_factory=list,
    )
        
    factor_training_recommendations: list[dict] = Field(
        default_factory=list,
    )
        
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
    )


class WorkflowState(BaseModel):
    """
    Mutable state passed through the orchestration workflow.
    """

    request: WorkflowRequest

    intent: str | None = None
    selected_tools: list[str] = Field(
        default_factory=list,
    )

    tool_results: dict[str, ToolResult] = Field(
        default_factory=dict,
    )

    errors: list[str] = Field(
        default_factory=list,
    )

    final_answer: str | None = None


class WorkflowResponse(BaseModel):
    """
    Final structured response returned by the workflow.
    """

    intent: str
    selected_tools: list[str]

    success: bool

    tool_results: dict[str, ToolResult]

    final_answer: str | None = None
    errors: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )