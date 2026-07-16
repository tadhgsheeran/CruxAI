from pathlib import Path

import torch
import torch.nn as nn


class MoonBoardMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(18 * 11, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        predictions = self.network(x)
        return predictions.squeeze(1)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "grade_predictor"
    / "moonboard_mlp_v1.pt"
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = MoonBoardMLP().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


model = load_model()


def predict_grade(holds: list[list[int]]) -> dict:
    route_tensor = torch.tensor(
        holds,
        dtype=torch.float32,
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(route_tensor).item()

    rounded_grade = round(prediction)

    return {
        "predicted_grade": prediction,
        "rounded_grade": rounded_grade,
        "formatted_grade": f"V{rounded_grade}",
        "model_version": "1.0.0",
    }
