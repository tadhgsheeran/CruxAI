from typing import Sequence

import numpy as np

from app.tools.schemas import ToolResult


def validate_route(
    route: Sequence[Sequence[int | float]],
) -> np.ndarray:
    """
    Validate and convert a route into an 18-by-11 binary array.
    """
    route_array = np.asarray(
        route,
        dtype=np.float32,
    )

    if route_array.shape != (18, 11):
        raise ValueError(
            "Route must have shape (18, 11). "
            f"Received {route_array.shape}."
        )

    if not np.all(
        np.isin(route_array, [0.0, 1.0])
    ):
        raise ValueError(
            "Route values must contain only 0s and 1s."
        )

    if route_array.sum() == 0:
        raise ValueError(
            "Route must contain at least one active hold."
        )

    return route_array


def calculate_move_distances(
    active_holds: np.ndarray,
) -> np.ndarray:
    """
    Estimate consecutive move distances.

    Holds are ordered from lowest to highest row and then
    from left to right within each row.
    """
    if len(active_holds) < 2:
        return np.asarray(
            [],
            dtype=np.float32,
        )

    ordered_holds = active_holds[
        np.lexsort(
            (
                active_holds[:, 1],
                active_holds[:, 0],
            )
        )
    ]

    differences = np.diff(
        ordered_holds,
        axis=0,
    )

    return np.sqrt(
        np.sum(
            differences ** 2,
            axis=1,
        )
    )


def describe_difficulty_factors(
    hold_count: int,
    vertical_span: int,
    horizontal_span: int,
    average_move_distance: float,
    maximum_move_distance: float,
    upper_board_ratio: float,
    center_hold_ratio: float,
) -> list[str]:
    """
    Produce deterministic descriptions from route features.
    """
    factors = []

    if hold_count <= 6:
        factors.append(
            "The route uses relatively few holds, which may "
            "create sustained difficulty with limited recovery."
        )
    elif hold_count >= 11:
        factors.append(
            "The route uses many holds, which may increase "
            "sequence complexity and power-endurance demands."
        )

    if vertical_span >= 14:
        factors.append(
            "The route covers most of the board vertically."
        )

    if horizontal_span >= 8:
        factors.append(
            "The route has a wide horizontal span, suggesting "
            "cross-body movement or lateral reaches."
        )

    if average_move_distance >= 3.5:
        factors.append(
            "The estimated average move distance is large."
        )

    if maximum_move_distance >= 5.0:
        factors.append(
            "The route contains at least one especially long "
            "estimated move."
        )

    if upper_board_ratio >= 0.50:
        factors.append(
            "A large share of the holds are located high on "
            "the board, which may create a difficult finish."
        )

    if center_hold_ratio <= 0.35:
        factors.append(
            "Many holds are positioned away from the board's "
            "center, which may increase body-positioning demands."
        )

    if not factors:
        factors.append(
            "The route has a relatively balanced hold layout "
            "without one extreme geometric feature."
        )

    return factors


def difficulty_analysis_tool(
    route: Sequence[Sequence[int | float]],
) -> ToolResult:
    """
    Calculate interpretable geometric features for a MoonBoard route.
    """
    try:
        route_array = validate_route(route)

        active_holds = np.argwhere(
            route_array == 1
        )

        rows = active_holds[:, 0]
        columns = active_holds[:, 1]

        hold_count = int(len(active_holds))

        minimum_row = int(rows.min())
        maximum_row = int(rows.max())
        minimum_column = int(columns.min())
        maximum_column = int(columns.max())

        vertical_span = int(
            maximum_row - minimum_row
        )

        horizontal_span = int(
            maximum_column - minimum_column
        )

        average_height = float(
            rows.mean()
        )

        board_coverage = float(
            hold_count / route_array.size
        )

        upper_board_ratio = float(
            np.mean(rows >= 9)
        )

        lower_board_ratio = float(
            np.mean(rows < 9)
        )

        center_hold_ratio = float(
            np.mean(
                (columns >= 3)
                & (columns <= 7)
            )
        )

        move_distances = calculate_move_distances(
            active_holds
        )

        if len(move_distances):
            average_move_distance = float(
                move_distances.mean()
            )

            maximum_move_distance = float(
                move_distances.max()
            )
        else:
            average_move_distance = 0.0
            maximum_move_distance = 0.0

        factors = describe_difficulty_factors(
            hold_count=hold_count,
            vertical_span=vertical_span,
            horizontal_span=horizontal_span,
            average_move_distance=(
                average_move_distance
            ),
            maximum_move_distance=(
                maximum_move_distance
            ),
            upper_board_ratio=upper_board_ratio,
            center_hold_ratio=center_hold_ratio,
        )

        return ToolResult(
            tool="difficulty_analysis",
            success=True,
            summary=(
                "Calculated geometric difficulty factors "
                "for the submitted route."
            ),
            data={
                "hold_count": hold_count,
                "active_holds": active_holds.tolist(),
                "minimum_row": minimum_row,
                "maximum_row": maximum_row,
                "minimum_column": minimum_column,
                "maximum_column": maximum_column,
                "vertical_span": vertical_span,
                "horizontal_span": horizontal_span,
                "average_height": average_height,
                "average_move_distance": (
                    average_move_distance
                ),
                "maximum_move_distance": (
                    maximum_move_distance
                ),
                "board_coverage": board_coverage,
                "upper_board_ratio": upper_board_ratio,
                "lower_board_ratio": lower_board_ratio,
                "center_hold_ratio": center_hold_ratio,
                "difficulty_factors": factors,
            },
        )

    except ValueError as exc:
        return ToolResult(
            tool="difficulty_analysis",
            success=False,
            summary=(
                "Route difficulty factors could not "
                "be calculated."
            ),
            error=str(exc),
        )

    except Exception as exc:
        return ToolResult(
            tool="difficulty_analysis",
            success=False,
            summary=(
                "An unexpected error occurred during "
                "route difficulty analysis."
            ),
            error=str(exc),
        )