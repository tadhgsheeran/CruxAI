from app.tools.factor_training import (
    build_factor_training_recommendations,
    identify_factor_keys,
)


def test_identifies_route_factor_keys():
    difficulty_data = {
        "hold_count": 10,
        "vertical_span": 14,
        "horizontal_span": 5,
        "average_move_distance": 3.54,
        "maximum_move_distance": 5.83,
    }

    keys = identify_factor_keys(
        difficulty_data
    )

    assert "vertical_span" in keys
    assert "large_average_move" in keys
    assert "long_maximum_move" in keys
    assert "wide_horizontal_span" not in keys


def test_builds_grounded_training_recommendations():
    difficulty_data = {
        "hold_count": 10,
        "vertical_span": 14,
        "horizontal_span": 5,
        "average_move_distance": 3.54,
        "maximum_move_distance": 5.83,
    }

    recommendations = (
        build_factor_training_recommendations(
            difficulty_data
        )
    )

    focuses = {
        item["focus"]
        for item in recommendations
    }

    assert "power endurance" in focuses
    assert (
        "dynamic movement and coordination"
        in focuses
    )

    combined_text = str(
        recommendations
    ).lower()

    assert "crimp" not in combined_text
    assert "sloper" not in combined_text
    assert "steep" not in combined_text