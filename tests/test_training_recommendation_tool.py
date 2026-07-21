from app.tools.training_recommendation import (
    build_training_query,
    training_recommendation_tool,
)


def test_build_training_query_adds_grade_context():
    query = build_training_query(
        question="What should I train for overhangs?",
        current_grade=5,
        target_grade=7,
    )

    assert "What should I train for overhangs?" in query
    assert "currently climbs around V5" in query
    assert "progress toward V7" in query


def test_training_recommendation_returns_success(
    monkeypatch,
):
    fake_results = [
        {
            "text": (
                "Body tension helps prevent the feet "
                "from cutting loose on overhangs."
            ),
            "source": "body_tension.md",
            "chunk_id": "body_tension_0",
            "score": 0.91,
        },
        {
            "text": (
                "Overhang climbing requires efficient "
                "hip positioning and core engagement."
            ),
            "source": "overhang_technique.md",
            "chunk_id": "overhang_technique_0",
            "score": 0.87,
        },
    ]

    def fake_search(query, top_k):
        return fake_results

    def fake_generate_answer(
        query,
        retrieved_results,
        max_new_tokens=250,
    ):
        return (
            "Train body tension and hip positioning "
            "[body_tension.md] "
            "[overhang_technique.md]"
        )

    monkeypatch.setattr(
        "app.tools.training_recommendation."
        "retrieval_service.search",
        fake_search,
    )

    monkeypatch.setattr(
        "app.tools.training_recommendation."
        "generation_service.generate_answer",
        fake_generate_answer,
    )

    result = training_recommendation_tool(
        question="What should I train for overhangs?",
        current_grade=5,
        target_grade=7,
        top_k=2,
    )

    assert result.success is True
    assert result.tool == "training_recommendation"
    assert result.data["current_grade"] == 5
    assert result.data["target_grade"] == 7
    assert result.data["result_count"] == 2
    assert "body tension" in (
        result.data["recommendation"].lower()
    )
    assert len(result.sources) == 2
    assert result.error is None


def test_training_recommendation_rejects_empty_question():
    result = training_recommendation_tool(
        question="   ",
    )

    assert result.success is False
    assert result.error is not None
    assert "empty" in result.error.lower()


def test_training_recommendation_rejects_invalid_top_k():
    result = training_recommendation_tool(
        question="How should I train?",
        top_k=0,
    )

    assert result.success is False
    assert result.error is not None
    assert "greater than 0" in result.error.lower()


def test_training_recommendation_rejects_negative_grade():
    result = training_recommendation_tool(
        question="How should I train?",
        current_grade=-1,
    )

    assert result.success is False
    assert result.error is not None
    assert "negative" in result.error.lower()
    
def test_build_training_query_adds_difficulty_factors():
    query = build_training_query(
        question="What should I train?",
        current_grade=5,
        target_grade=7,
        difficulty_factors=[
            "The estimated average move distance is large.",
            "The route has a wide horizontal span.",
        ],
    )

    assert "average move distance" in query
    assert "wide horizontal span" in query
    
def test_training_query_does_not_invent_hold_types():
    query = build_training_query(
        question="What should I train?",
        current_grade=5,
        target_grade=6,
        difficulty_factors=[
            "The route covers most of the board vertically.",
            "The estimated average move distance is large.",
        ],
    )

    assert "crimp" not in query.lower()
    assert "sloper" not in query.lower()
    assert "small edge" not in query.lower()