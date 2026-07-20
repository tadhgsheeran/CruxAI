from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from app.tools.schemas import ToolResult


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CLIMBS_PATH = PROJECT_ROOT / "data" / "climbs.txt"
GRADES_PATH = PROJECT_ROOT / "data" / "grades.txt"


@lru_cache(maxsize=1)
def load_route_database() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the MoonBoard routes and grades from disk.

    Routes are stored as consecutive 18-by-11 grids in climbs.txt.
    """
    climbs = np.loadtxt(
        CLIMBS_PATH,
        dtype=np.float32,
    )

    grades = np.loadtxt(
        GRADES_PATH,
        dtype=np.int64,
    )

    if climbs.shape[1] != 11:
        raise ValueError(
            "Each route row must contain exactly 11 columns."
        )

    if climbs.shape[0] % 18 != 0:
        raise ValueError(
            "The number of climb rows must be divisible by 18."
        )

    routes = climbs.reshape(-1, 18, 11)

    if len(routes) != len(grades):
        raise ValueError(
            "The number of routes does not match "
            "the number of grades."
        )

    return routes, grades


def validate_query_route(
    route: Sequence[Sequence[int | float]],
) -> np.ndarray:
    """
    Validate and convert a query route into an 18-by-11 array.
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

    if not np.all(np.isin(route_array, [0.0, 1.0])):
        raise ValueError(
            "Route values must contain only 0s and 1s."
        )

    if route_array.sum() == 0:
        raise ValueError(
            "Route must contain at least one active hold."
        )

    return route_array


def calculate_jaccard_similarities(
    query_route: np.ndarray,
    routes: np.ndarray,
) -> np.ndarray:
    """
    Calculate Jaccard similarity between one route and all routes.
    """
    query_flat = query_route.reshape(1, -1).astype(bool)
    routes_flat = routes.reshape(len(routes), -1).astype(bool)

    intersections = np.logical_and(
        routes_flat,
        query_flat,
    ).sum(axis=1)

    unions = np.logical_or(
        routes_flat,
        query_flat,
    ).sum(axis=1)

    similarities = np.divide(
        intersections,
        unions,
        out=np.zeros_like(
            intersections,
            dtype=np.float64,
        ),
        where=unions != 0,
    )

    return similarities


def similar_route_search_tool(
    route: Sequence[Sequence[int | float]],
    top_k: int = 5,
    exclude_exact_match: bool = True,
) -> ToolResult:
    """
    Find routes with similar active hold positions.
    """
    try:
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_route = validate_query_route(route)

        routes, grades = load_route_database()

        similarities = calculate_jaccard_similarities(
            query_route=query_route,
            routes=routes,
        )

        if exclude_exact_match:
            exact_matches = np.all(
                routes == query_route,
                axis=(1, 2),
            )

            similarities[exact_matches] = -1.0

        ranked_indices = np.argsort(
            similarities,
        )[::-1]

        selected_indices = ranked_indices[:top_k]

        matches = []

        for route_index in selected_indices:
            database_route = routes[route_index]

            active_holds = np.argwhere(
                database_route == 1
            ).tolist()

            matches.append(
                {
                    "route_id": int(route_index),
                    "grade": int(grades[route_index]),
                    "formatted_grade": (
                        f"V{int(grades[route_index])}"
                    ),
                    "similarity": float(
                        similarities[route_index]
                    ),
                    "active_holds": active_holds,
                }
            )

        return ToolResult(
            tool="similar_route_search",
            success=True,
            summary=(
                f"Found {len(matches)} similar "
                "MoonBoard routes."
            ),
            data={
                "top_k": top_k,
                "similarity_metric": "jaccard",
                "matches": matches,
            },
        )

    except (ValueError, FileNotFoundError) as exc:
        return ToolResult(
            tool="similar_route_search",
            success=False,
            summary="Similar routes could not be found.",
            data={
                "top_k": top_k,
            },
            error=str(exc),
        )

    except Exception as exc:
        return ToolResult(
            tool="similar_route_search",
            success=False,
            summary=(
                "An unexpected error occurred during "
                "similar-route search."
            ),
            data={
                "top_k": top_k,
            },
            error=str(exc),
        )