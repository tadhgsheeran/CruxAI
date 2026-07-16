from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from models.grade_predictor.architecture import MoonBoardMLP


class MoonBoardPredictor:
    """
    Loads the trained MoonBoard model and produces grade predictions.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.device = self._select_device()

        if model_path is None:
            model_path = Path(__file__).parent / "moonboard_mlp_v1.pt"

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}"
            )

        self.checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,

        )

        self.model = MoonBoardMLP().to(self.device)

        self.model.load_state_dict(
            self.checkpoint["model_state_dict"]
        )

        self.model.eval()

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    @staticmethod
    def _validate_route(
        route: Sequence[Sequence[int | float]],
    ) -> np.ndarray:
        route_array = np.asarray(
            route,
            dtype=np.float32,
        )

        if route_array.shape != (18, 11):
            raise ValueError(
                "Route must have shape (18, 11). "
                f"Received {route_array.shape}."
            )

        if not np.all(
            np.isin(route_array, [0.0, 1.0])
        ):
            raise ValueError(
                "Route values must contain only 0s and 1s."
            )

        if route_array.sum() == 0:
            raise ValueError(
                "Route must contain at least one active hold."
            )

        return route_array

    def predict(
        self,
        route: Sequence[Sequence[int | float]],
    ) -> dict[str, object]:
        route_array = self._validate_route(route)

        route_tensor = torch.tensor(
            route_array,
            dtype=torch.float32,
        )

        # Add a batch dimension:
        # [18, 11] becomes [1, 18, 11]
        route_tensor = route_tensor.unsqueeze(0)
        route_tensor = route_tensor.to(self.device)

        with torch.no_grad():
            prediction = self.model(route_tensor)

        predicted_grade = float(
            prediction.cpu().item()
        )

        rounded_grade = int(
            np.rint(predicted_grade)
        )

        return {
            "predicted_grade": predicted_grade,
            "rounded_grade": rounded_grade,
            "formatted_grade": f"V{rounded_grade}",
            "model_version": self.checkpoint.get(
                "model_version",
                "unknown",
            ),
        }
