from pydantic import BaseModel, Field, field_validator


class RouteInput(BaseModel):
    holds: list[list[int]] = Field(
        ...,
        description="An 18-row by 11-column MoonBoard route matrix.",
    )

    @field_validator("holds")
    @classmethod
    def validate_route_shape(cls, holds):
        if len(holds) != 18:
            raise ValueError(
                "Route matrix must contain exactly 18 rows."
            )

        for row in holds:
            if len(row) != 11:
                raise ValueError(
                    "Each route matrix row must contain exactly 11 values."
                )

        return holds


class GradePrediction(BaseModel):
    predicted_grade: float
    rounded_grade: int
    formatted_grade: str
    model_version: str

        
        
class RetrievalRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="The climbing question to search for.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of relevant chunks to return.",
    )


class RetrievalResult(BaseModel):
    text: str
    source: str
    chunk_id: str
    score: float


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievalResult]
        
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
        
from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The user's climbing or route-analysis request.",
    )


class RouteDecisionResponse(BaseModel):
    intent: str
    tools: list[str]
    reason: str

from typing import Any

from pydantic import BaseModel, Field

from app.tools.schemas import ToolResult


class AnalyzeRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The user's climbing or route-analysis request.",
    )

    route: list[list[int | float]] | None = None

    current_grade: int | None = Field(
        default=None,
        ge=0,
    )

    target_grade: int | None = Field(
        default=None,
        ge=0,
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
    )


class AnalyzeResponse(BaseModel):
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