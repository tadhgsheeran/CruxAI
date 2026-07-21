import numpy as np

from app.tools.difficulty_analysis import (
    calculate_move_distances,
    difficulty_analysis_tool,
)


def make_test_route() -> list[list[int]]:
    route = np.zeros(
        (18, 11),
        dtype=int,
    )

    route[0, 2] = 1
    route[3, 4] = 1
    route[7, 8] = 1
    route[12, 5] = 1
    route[17, 9] = 1

    return route.tolist()


def test_difficulty_analysis_returns_features():
    result = difficulty_analysis_tool(
        make_test_route()
    )

    assert result.success is True
    assert result.tool == "difficulty_analysis"

    assert result.data["hold_count"] == 5
    assert result.data["vertical_span"] == 17
    assert result.data["horizontal_span"] == 7

    assert (
        result.data["average_move_distance"]
        > 0
    )

    assert (
        result.data["maximum_move_distance"]
        > 0
    )

    assert result.data["difficulty_factors"]
    assert result.error is None


def test_difficulty_analysis_returns_active_holds():
    result = difficulty_analysis_tool(
        make_test_route()
    )

    assert result.success is True
    assert len(result.data["active_holds"]) == 5
    assert [0, 2] in result.data["active_holds"]
    assert [17, 9] in result.data["active_holds"]


def test_difficulty_analysis_rejects_wrong_shape():
    result = difficulty_analysis_tool(
        [[0, 1], [1, 0]]
    )

    assert result.success is False
    assert result.error is not None
    assert "shape" in result.error.lower()


def test_difficulty_analysis_rejects_empty_route():
    route = np.zeros(
        (18, 11),
        dtype=int,
    ).tolist()

    result = difficulty_analysis_tool(route)

    assert result.success is False
    assert result.error is not None
    assert "active hold" in result.error.lower()


def test_difficulty_analysis_rejects_nonbinary_route():
    route = np.zeros(
        (18, 11),
        dtype=float,
    )

    route[4, 5] = 0.5

    result = difficulty_analysis_tool(
        route.tolist()
    )

    assert result.success is False
    assert result.error is not None
    assert "0s and 1s" in result.error


def test_calculate_move_distances_orders_bottom_to_top():
    active_holds = np.asarray(
        [
            [6, 4],
            [0, 0],
            [3, 0],
        ],
        dtype=np.float32,
    )

    distances = calculate_move_distances(
        active_holds
    )

    assert len(distances) == 2

    assert np.isclose(
        distances[0],
        3.0,
    )

    assert np.isclose(
        distances[1],
        5.0,
    )