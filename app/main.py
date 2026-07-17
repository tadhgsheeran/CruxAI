from fastapi import FastAPI

from app.model import predict_grade
from app.retrieval.service import retrieval_service
from app.schemas import (
    GradePrediction,
    RetrievalRequest,
    RetrievalResponse,
    RouteInput,
)

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