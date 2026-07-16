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