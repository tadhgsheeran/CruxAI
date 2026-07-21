import evaluation.evaluate_router as evaluator

from app.orchestration.hybrid_router import (
    hybrid_route_request,
)


def evaluate_hybrid_router(
    verbose: bool = True,
) -> dict:
    """
    Evaluate the hybrid router using the standard
    router benchmark and metric calculations.
    """
    original_route_request = evaluator.route_request

    try:
        evaluator.route_request = (
            hybrid_route_request
        )

        return evaluator.evaluate_router(
            verbose=verbose
        )

    finally:
        evaluator.route_request = (
            original_route_request
        )


if __name__ == "__main__":
    evaluate_hybrid_router()