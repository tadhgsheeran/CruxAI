from functools import lru_cache
from typing import Sequence

from models.grade_predictor.predictor import MoonBoardPredictor

from app.tools.schemas import ToolResult


@lru_cache(maxsize=1)
def get_predictor() -> MoonBoardPredictor:
    """
    Load and cache one MoonBoard predictor instance.

    The first call loads the PyTorch model. Later calls reuse it
    instead of loading the checkpoint again.
    """
    return MoonBoardPredictor()


def predict_grade_tool(
    route: Sequence[Sequence[int | float]],
) -> ToolResult:
    """
    Predict the difficulty grade of an 18-by-11 MoonBoard route.
    """
    try:
        predictor = get_predictor()
        prediction = predictor.predict(route)

        return ToolResult(
            tool="grade_prediction",
            success=True,
            summary=(
                "The route was predicted as "
                f"{prediction['formatted_grade']}."
            ),
            data=prediction,
        )

    except (ValueError, FileNotFoundError) as exc:
        return ToolResult(
            tool="grade_prediction",
            success=False,
            summary="The route grade could not be predicted.",
            error=str(exc),
        )

    except Exception as exc:
        return ToolResult(
            tool="grade_prediction",
            success=False,
            summary=(
                "An unexpected error occurred during "
                "grade prediction."
            ),
            error=str(exc),
        )