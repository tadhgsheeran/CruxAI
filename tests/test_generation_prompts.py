import pytest

from app.prompts.generation_prompts import (
    build_user_prompt,
    get_system_prompt,
)


def test_baseline_prompt_is_available():
    prompt = get_system_prompt(
        "baseline"
    )

    assert "climbing assistant" in prompt.lower()


def test_grounded_prompt_requires_context():
    prompt = get_system_prompt(
        "evidence_grounded"
    )

    assert "only the supplied context" in prompt
    assert "do not invent" in prompt.lower()
    assert "square brackets" in prompt.lower()


def test_invalid_prompt_version_is_rejected():
    with pytest.raises(
        ValueError,
        match="prompt_version",
    ):
        get_system_prompt("invalid")


def test_build_user_prompt_includes_query_and_context():
    prompt = build_user_prompt(
        query="How do heel hooks work?",
        context="Heel hooks use the heel to pull.",
    )

    assert "How do heel hooks work?" in prompt
    assert "Heel hooks use the heel" in prompt