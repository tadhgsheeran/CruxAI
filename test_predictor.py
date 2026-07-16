import numpy as np

from models.grade_predictor.predictor import MoonBoardPredictor


def main() -> None:
    predictor = MoonBoardPredictor()

    test_route = np.load(
        "data/example_test_route.npy"
    )

    result = predictor.predict(test_route)

    print("Device:", predictor.device)
    print("Standalone prediction:", result)


if __name__ == "__main__":
    main()
