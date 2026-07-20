import re

import torch

from app.generation.service import generation_service
from app.orchestration.router import Intent, RoutingDecision


LABEL_TO_DECISION = {
    "GENERAL": RoutingDecision(
        intent=Intent.GENERAL_CLIMBING_QUESTION,
        tools=["knowledge_retrieval"],
        reason="The language model classified this as climbing knowledge.",
    ),
    "GRADE": RoutingDecision(
        intent=Intent.ROUTE_GRADE_PREDICTION,
        tools=["grade_prediction"],
        reason="The language model classified this as grade prediction.",
    ),
    "SIMILAR": RoutingDecision(
        intent=Intent.SIMILAR_ROUTE_SEARCH,
        tools=["similar_route_search"],
        reason="The language model classified this as similar-route search.",
    ),
    "TRAINING": RoutingDecision(
        intent=Intent.TRAINING_RECOMMENDATION,
        tools=["training_recommendation"],
        reason="The language model classified this as training advice.",
    ),
    "GRADE_SIMILAR": RoutingDecision(
        intent=Intent.MULTI_STEP_ANALYSIS,
        tools=[
            "grade_prediction",
            "similar_route_search",
        ],
        reason=(
            "The language model detected grade prediction "
            "and similar-route search."
        ),
    ),
    "GRADE_TRAINING": RoutingDecision(
        intent=Intent.MULTI_STEP_ANALYSIS,
        tools=[
            "grade_prediction",
            "training_recommendation",
        ],
        reason=(
            "The language model detected grade prediction "
            "and training advice."
        ),
    ),
    "SIMILAR_TRAINING": RoutingDecision(
        intent=Intent.MULTI_STEP_ANALYSIS,
        tools=[
            "similar_route_search",
            "training_recommendation",
        ],
        reason=(
            "The language model detected similar-route search "
            "and training advice."
        ),
    ),
    "ALL_THREE": RoutingDecision(
        intent=Intent.MULTI_STEP_ANALYSIS,
        tools=[
            "grade_prediction",
            "similar_route_search",
            "training_recommendation",
        ],
        reason="The language model detected all three route tools.",
    ),
    "UNSUPPORTED": RoutingDecision(
        intent=Intent.UNSUPPORTED_REQUEST,
        tools=[],
        reason="The language model classified this as unsupported.",
    ),
}


ROUTER_SYSTEM_PROMPT = """
You classify requests for CruxAI, a climbing assistant.

Choose exactly one label:

GENERAL
Climbing terminology, technique, movement, or knowledge.

GRADE
Predict, estimate, rate, score, or evaluate a route's difficulty.

SIMILAR
Find routes or hold layouts resembling another route.

TRAINING
Exercises, drills, workouts, plans, preparation, or weaknesses.

GRADE_SIMILAR
Both route-grade prediction and similar-route search.

GRADE_TRAINING
Both route-grade prediction and training recommendations.

SIMILAR_TRAINING
Both similar-route search and training recommendations.

ALL_THREE
Grade prediction, similar-route search, and training advice.

UNSUPPORTED
Anything unrelated to climbing or route analysis.

Examples:

Question: Explain how a heel hook works.
Label: GENERAL

Question: Estimate this route's grade.
Label: GRADE

Question: Find routes resembling this setup.
Label: SIMILAR

Question: Give me drills for crimps.
Label: TRAINING

Question: Score this route and find comparable problems.
Label: GRADE_SIMILAR

Question: How hard is this and what should I train?
Label: GRADE_TRAINING

Question: Find similar climbs and build a workout.
Label: SIMILAR_TRAINING

Question: Rate this route, find matches, and recommend exercises.
Label: ALL_THREE

Question: Help me write an email.
Label: UNSUPPORTED

Return only the label.
"""


def extract_label(text: str) -> str:
    """
    Extract one allowed routing label from model output.
    """
    normalized = text.strip().upper()

    normalized = re.sub(
        r"[^A-Z_]",
        " ",
        normalized,
    )

    tokens = normalized.split()

    # Check longer labels first.
    ordered_labels = [
        "SIMILAR_TRAINING",
        "GRADE_TRAINING",
        "GRADE_SIMILAR",
        "ALL_THREE",
        "UNSUPPORTED",
        "TRAINING",
        "SIMILAR",
        "GENERAL",
        "GRADE",
    ]

    compact_text = "_".join(tokens)

    for label in ordered_labels:
        if label in compact_text:
            return label

    raise ValueError(
        f"No valid routing label found in output: {text!r}"
    )


def llm_route_request(
    question: str,
    max_new_tokens: int = 12,
) -> RoutingDecision:
    """
    Route a request using label-only local LLM classification.
    """
    normalized_question = question.strip()

    if not normalized_question:
        return LABEL_TO_DECISION["UNSUPPORTED"].model_copy(
            update={"reason": "The request is empty."}
        )

    try:
        generation_service._load_model()

        tokenizer = generation_service.tokenizer
        model = generation_service.model

        messages = [
            {
                "role": "system",
                "content": ROUTER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Question: {normalized_question}\n"
                    "Label:"
                ),
            },
        ]

        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated_ids[
            0,
            model_inputs["input_ids"].shape[1] :,
        ]

        response_text = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        label = extract_label(response_text)

        return LABEL_TO_DECISION[label]

    except Exception as exc:
        return RoutingDecision(
            intent=Intent.UNSUPPORTED_REQUEST,
            tools=[],
            reason=(
                "The label-based language-model router failed: "
                f"{exc}"
            ),
        )