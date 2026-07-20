import evaluation.evaluate_router as evaluator

from app.orchestration.semantic_router import (
    semantic_route_request,
)


evaluator.route_request = semantic_route_request


if __name__ == "__main__":
    evaluator.evaluate_router()