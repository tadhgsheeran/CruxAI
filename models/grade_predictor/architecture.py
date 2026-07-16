import torch
import torch.nn as nn


class MoonBoardMLP(nn.Module):
    """
    Predicts a numerical MoonBoard V-grade from an 18 x 11
    binary route grid.
    """

    def __init__(self) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = self.network(x)
        return predictions.squeeze(1)