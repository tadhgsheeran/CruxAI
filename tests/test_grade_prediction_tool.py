import numpy as np

from app.tools.grade_prediction import predict_grade_tool


def make_test_route() -> list[list[int]]:
    route = np.zeros((18, 11), dtype=int)

    route[1, 2] = 1
    route[4, 4] = 1
    route[7, 6] = 1
    route[10, 5] = 1
    route[13, 7] = 1
    route[16, 8] = 1

    return route.tolist()


def test_predict_grade_tool_returns_success():
    result = predict_grade_tool(make_test_route())

    assert result.success is True
    assert result.tool == "grade_prediction"
    assert "predicted_grade" in result.data
    assert "rounded_grade" in result.data
    assert "formatted_grade" in result.data
    assert result.error is None


def test_predict_grade_tool_rejects_wrong_shape():
    invalid_route = [[0, 1], [1, 0]]

    result = predict_grade_tool(invalid_route)

    assert result.success is False
    assert result.tool == "grade_prediction"
    assert result.error is not None
    assert "shape" in result.error.lower()


def test_predict_grade_tool_rejects_empty_route():
    empty_route = np.zeros((18, 11), dtype=int).tolist()

    result = predict_grade_tool(empty_route)

    assert result.success is False
    assert result.error is not None
    assert "at least one active hold" in result.error.lower()