import pytest

from evaluation.evaluate_quantization import (
    load_model,
)


def test_load_model_rejects_invalid_mode():
    with pytest.raises(
        ValueError,
        match="mode",
    ):
        load_model("invalid")