from fastapi import FastAPI

from app.model import predict_grade
from app.schemas import GradePrediction, RouteInput


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