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