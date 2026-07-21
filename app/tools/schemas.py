from typing import Any, Literal

from pydantic import BaseModel, Field


ToolName = Literal[
    "grade_prediction",
    "difficulty_analysis",
    "knowledge_retrieval",
    "similar_route_search",
    "training_recommendation",
]


class ToolSource(BaseModel):
    document: str
    chunk_id: str | None = None
    score: float | None = None
    text: str | None = None


class ToolResult(BaseModel):
    tool: ToolName
    success: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    sources: list[ToolSource] = Field(default_factory=list)
    error: str | None = None