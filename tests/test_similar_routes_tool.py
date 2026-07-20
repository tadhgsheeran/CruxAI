import numpy as np

from app.tools.similar_routes import (
    load_route_database,
    similar_route_search_tool,
)


def make_test_route() -> list[list[int]]:
    routes, _ = load_route_database()

    return routes[0].astype(int).tolist()


def test_route_database_loads_correctly():
    routes, grades = load_route_database()

    assert routes.shape == (11906, 18, 11)
    assert grades.shape == (11906,)
    assert len(routes) == len(grades)


def test_similar_route_search_returns_matches():
    result = similar_route_search_tool(
        route=make_test_route(),
        top_k=5,
    )

    assert result.success is True
    assert result.tool == "similar_route_search"
    assert result.data["similarity_metric"] == "jaccard"
    assert len(result.data["matches"]) == 5
    assert result.error is None

    for match in result.data["matches"]:
        assert "route_id" in match
        assert "grade" in match
        assert "formatted_grade" in match
        assert "similarity" in match
        assert "active_holds" in match

        assert 0.0 <= match["similarity"] <= 1.0


def test_similar_route_search_rejects_wrong_shape():
    invalid_route = [[0, 1], [1, 0]]

    result = similar_route_search_tool(
        route=invalid_route,
        top_k=5,
    )

    assert result.success is False
    assert result.error is not None
    assert "shape" in result.error.lower()


def test_similar_route_search_rejects_invalid_top_k():
    result = similar_route_search_tool(
        route=make_test_route(),
        top_k=0,
    )

    assert result.success is False
    assert result.error is not None
    assert "greater than 0" in result.error.lower()