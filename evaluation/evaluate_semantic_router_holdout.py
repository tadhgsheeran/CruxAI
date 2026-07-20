from pathlib import Path

import evaluation.evaluate_router as evaluator

from app.orchestration.semantic_router import (
    semantic_route_request,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

evaluator.BENCHMARK_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "router_holdout.jsonl"
)

evaluator.route_request = semantic_route_request


if __name__ == "__main__":
    evaluator.evaluate_router()