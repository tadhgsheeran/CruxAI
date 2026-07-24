from fastapi import FastAPI

from app.model import predict_grade

from app.generation.service import generation_service
from app.retrieval.service import retrieval_service
from app.schemas import (
    AskRequest,
    AskResponse,
    GradePrediction,
    RetrievalRequest,
    RetrievalResponse,
    RouteInput,
    RouteDecisionResponse,
    RouteRequest,
    AnalyzeRequest,
    AnalyzeResponse,
)

from app.orchestration.semantic_router import (
    semantic_route_request,
)

from app.orchestration.workflow import run_workflow

from app.orchestration.workflow_schemas import (
    WorkflowRequest,
)

from app.orchestration.router import (
    Intent,
    route_request,
)

MIN_RETRIEVAL_SCORE = 0.35

app = FastAPI(
    title="CruxAI API",
    description="API for predicting MoonBoard climbing grades.",
    version="1.0.0",
)

@app.get("/")
def root():
    return {
        "message": "CruxAI API is running."
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "CruxAI",
    }


@app.post(
    "/predict-grade",
    response_model=GradePrediction,
)
def predict_route_grade(route: RouteInput):
    return predict_grade(route.holds)

@app.post(
    "/retrieve",
    response_model=RetrievalResponse,
)
def retrieve_climbing_knowledge(request: RetrievalRequest):
    results = retrieval_service.search(
        query=request.query,
        top_k=request.top_k,
    )

    return {
        "query": request.query,
        "results": results,
    }

@app.post(
    "/ask",
    response_model=AskResponse,
)
async def ask_question(
    request: AskRequest,
) -> AskResponse:
    routing_decision = route_request(
        request.query
    )

    if (
        routing_decision.intent
        == Intent.UNSUPPORTED_REQUEST
    ):
        return AskResponse(
            query=request.query,
            answer=(
                "I do not have enough relevant information "
                "in the climbing knowledge base to answer "
                "that question."
            ),
            sources=[],
        )

    retrieved_results = retrieval_service.search(
        query=request.query,
        top_k=request.top_k,
    )

    answer = generation_service.generate_answer(
        query=request.query,
        retrieved_results=retrieved_results,
    )

    source_names = list(
        dict.fromkeys(
            result["source"]
            for result in retrieved_results
        )
    )

    return AskResponse(
        query=request.query,
        answer=answer,
        sources=source_names,
    )

@app.post(
    "/route",
    response_model=RouteDecisionResponse,
)
def route_user_request(
    request: RouteRequest,
) -> RouteDecisionResponse:
    decision = semantic_route_request(
        request.question
    )

    return RouteDecisionResponse(
        intent=decision.intent.value,
        tools=decision.tools,
        reason=decision.reason,
    )

@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
)
async def analyze_request(
    request: AnalyzeRequest,
) -> AnalyzeResponse:
    workflow_request = WorkflowRequest(
        question=request.question,
        route=request.route,
        current_grade=request.current_grade,
        target_grade=request.target_grade,
        top_k=request.top_k,
    )

    result = run_workflow(
        workflow_request,
    )

    return AnalyzeResponse(
        intent=result.intent,
        selected_tools=result.selected_tools,
        success=result.success,
        tool_results=result.tool_results,
        final_answer=result.final_answer,
        errors=result.errors,
        metadata=result.metadata,
    )