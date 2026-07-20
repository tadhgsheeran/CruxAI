from pathlib import Path

import evaluation.evaluate_router as evaluator


PROJECT_ROOT = Path(__file__).resolve().parent.parent

evaluator.BENCHMARK_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "router_holdout.jsonl"
)


if __name__ == "__main__":
    evaluator.evaluate_router()