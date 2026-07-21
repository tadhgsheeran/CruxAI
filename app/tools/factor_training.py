FACTOR_TRAINING_MAP = {
    "vertical_span": {
        "focus": "power endurance",
        "reason": (
            "A route covering much of the board may require "
            "sustained movement across a longer sequence."
        ),
        "methods": [
            "link short hard sequences",
            "repeat submaximal circuits",
            "practice maintaining technique while fatigued",
        ],
    },
    "large_average_move": {
        "focus": "dynamic movement and coordination",
        "reason": (
            "Larger estimated move distances may require more "
            "controlled acceleration and accurate movement."
        ),
        "methods": [
            "practice controlled deadpoints",
            "use progressive reach drills",
            "repeat moves while emphasizing accurate foot placement",
        ],
    },
    "long_maximum_move": {
        "focus": "explosive pulling and lock-off control",
        "reason": (
            "An especially long estimated move may demand power "
            "and control through an extended range."
        ),
        "methods": [
            "practice assisted long-reach moves",
            "train controlled lock-offs",
            "use low-volume explosive pulling drills",
        ],
    },
    "wide_horizontal_span": {
        "focus": "cross-body positioning and lateral control",
        "reason": (
            "A wide horizontal span may increase lateral movement "
            "and body-positioning demands."
        ),
        "methods": [
            "practice flagging",
            "use cross-body movement drills",
            "train shoulder and trunk stability",
        ],
    },
    "few_holds": {
        "focus": "movement efficiency and recovery",
        "reason": (
            "A route with few holds may provide limited opportunities "
            "to adjust position or recover."
        ),
        "methods": [
            "practice efficient sequencing",
            "reduce unnecessary grip tension",
            "rehearse rests and pacing",
        ],
    },
}


def identify_factor_keys(
    difficulty_data: dict,
) -> list[str]:
    keys = []

    if difficulty_data.get("vertical_span", 0) >= 14:
        keys.append("vertical_span")

    if (
        difficulty_data.get(
            "average_move_distance",
            0,
        )
        >= 3.5
    ):
        keys.append("large_average_move")

    if (
        difficulty_data.get(
            "maximum_move_distance",
            0,
        )
        >= 5.0
    ):
        keys.append("long_maximum_move")

    if difficulty_data.get("horizontal_span", 0) >= 8:
        keys.append("wide_horizontal_span")

    if difficulty_data.get("hold_count", 100) <= 6:
        keys.append("few_holds")

    return keys


def build_factor_training_recommendations(
    difficulty_data: dict,
) -> list[dict]:
    factor_keys = identify_factor_keys(
        difficulty_data
    )

    return [
        {
            "factor": factor_key,
            **FACTOR_TRAINING_MAP[factor_key],
        }
        for factor_key in factor_keys
    ]