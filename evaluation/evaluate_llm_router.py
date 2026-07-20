import evaluation.evaluate_router as evaluator

from app.orchestration.llm_router import llm_route_request


evaluator.route_request = llm_route_request


if __name__ == "__main__":
    evaluator.evaluate_router()