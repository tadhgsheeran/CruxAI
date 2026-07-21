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
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
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

@app.post("/ask", response_model=AskResponse)
def ask_cruxai(request: AskRequest):
    results = retrieval_service.search(
        query=request.query,
        top_k=request.top_k,
    )

    if not results or results[0]["score"] < MIN_RETRIEVAL_SCORE:
        return {
            "query": request.query,
            "answer": (
                "I do not have enough relevant information in the "
                "CruxAI knowledge base to answer that question."
            ),
            "sources": [],
        }

    answer = generation_service.generate_answer(
        query=request.query,
        retrieved_results=results,
    )

    sources = [result["source"] for result in results]

    return {
        "query": request.query,
        "answer": answer,
        "sources": sources,
    }

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