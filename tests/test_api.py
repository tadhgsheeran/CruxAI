import numpy as np
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "CruxAI API is running."
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
    }


def test_predict_grade_with_real_route():
    route = np.load("data/example_test_route.npy")

    response = client.post(
        "/predict-grade",
        json={
            "holds": route.astype(int).tolist()
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert "predicted_grade" in result
    assert "rounded_grade" in result
    assert "formatted_grade" in result
    assert "model_version" in result

    assert abs(result["predicted_grade"] - 6.1741242) < 0.001
    assert result["rounded_grade"] == 6
    assert result["formatted_grade"] == "V6"
    assert result["model_version"] == "1.0.0"


def test_rejects_wrong_number_of_rows():
    invalid_route = [
        [0] * 11
        for _ in range(17)
    ]

    response = client.post(
        "/predict-grade",
        json={"holds": invalid_route},
    )

    assert response.status_code == 422


def test_rejects_wrong_number_of_columns():
    invalid_route = [
        [0] * 10
        for _ in range(18)
    ]

    response = client.post(
        "/predict-grade",
        json={"holds": invalid_route},
    )

    assert response.status_code == 422