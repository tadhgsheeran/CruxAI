import evaluation.evaluate_router as evaluator

from app.orchestration.hybrid_router import (
    hybrid_route_request,
)


evaluator.route_request = hybrid_route_request


if __name__ == "__main__":
    evaluator.evaluate_router()