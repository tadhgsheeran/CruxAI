from app.generation.service import generation_service
from app.retrieval.service import retrieval_service
from app.tools.schemas import ToolResult, ToolSource

import os

def build_training_query(
    question: str,
    current_grade: int | None = None,
    target_grade: int | None = None,
    difficulty_factors: list[str] | None = None,
) -> str:
    """
    Add optional climber and route context to the retrieval query.
    """
    query_parts = [question.strip()]

    if current_grade is not None:
        query_parts.append(
            f"The climber currently climbs around V{current_grade}."
        )

    if target_grade is not None:
        query_parts.append(
            f"The climber wants to progress toward V{target_grade}."
        )

    if difficulty_factors:
        factor_text = "; ".join(
            difficulty_factors
        )

        query_parts.append(
            "The route difficulty analysis identified: "
            f"{factor_text}"
        )

    return " ".join(query_parts)

def build_deterministic_recommendation(
    retrieved_results: list[dict],
) -> str:
    """
    Format retrieved evidence into a concise, cited training plan.
    """
    sections = []

    for result in retrieved_results:
        source = result["source"]
        text = result["text"].strip()

        cleaned_lines = []

        for line in text.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.lower() in {
                "## safety",
                "## training and practice",
                "## when it is useful",
                "## common mistakes",
            }:
                continue

            if stripped.startswith("#"):
                continue

            cleaned_lines.append(stripped)

        cleaned_text = " ".join(cleaned_lines)

        if not cleaned_text:
            continue

        sections.append(
            f"### {source.removesuffix('.md').replace('_', ' ').title()}\n"
            f"{cleaned_text}\n"
            f"Source: [{source}]"
        )

    if not sections:
        return (
            "The retrieved evidence did not contain enough "
            "information to create a training recommendation."
        )

    return (
        "## Recommended Training Plan\n\n"
        "Based on the retrieved climbing evidence, focus on the "
        "following areas:\n\n"
        + "\n\n".join(sections)
    )

def training_recommendation_tool(
    question: str,
    current_grade: int | None = None,
    target_grade: int | None = None,
    difficulty_factors: list[str] | None = None,
    factor_training_recommendations: list[dict] | None = None,
    top_k: int = 3,
) -> ToolResult:
    """
    Generate an evidence-grounded climbing training recommendation.
    """
    try:
        if not question.strip():
            raise ValueError(
                "Training question cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if current_grade is not None and current_grade < 0:
            raise ValueError(
                "current_grade cannot be negative."
            )

        if target_grade is not None and target_grade < 0:
            raise ValueError(
                "target_grade cannot be negative."
            )

        retrieval_query = build_training_query(
            question=question,
            current_grade=current_grade,
            target_grade=target_grade,
            difficulty_factors=difficulty_factors,
        )

        if factor_training_recommendations:
            grounded_context = []

            for item in factor_training_recommendations:
                methods = ", ".join(
                    item.get("methods", [])
                )

                grounded_context.append(
                    f"Training focus: {item.get('focus')}. "
                    f"Reason: {item.get('reason')} "
                    f"Suggested methods: {methods}."
                )

            retrieval_query += (
                " Grounded route-specific training guidance: "
                + " ".join(grounded_context)
            )
        
        retrieved_results = retrieval_service.search(
            query=retrieval_query,
            top_k=top_k,
        )

        generation_mode = os.getenv(
            "CRUXAI_GENERATION_MODE",
            "llm",
        )

        if generation_mode == "deterministic":
            recommendation = build_deterministic_recommendation(
                retrieved_results
            )
        else:
            recommendation = generation_service.generate_answer(
                query=retrieval_query,
                retrieved_results=retrieved_results,
                max_new_tokens=160,
            )
                

        sources = [
            ToolSource(
                document=result["source"],
                chunk_id=result["chunk_id"],
                score=result["score"],
                text=result["text"],
            )
            for result in retrieved_results
        ]

        return ToolResult(
            tool="training_recommendation",
            success=True,
            summary=(
                "Generated an evidence-grounded "
                "climbing training recommendation."
            ),
            data={
                "question": question,
                "retrieval_query": retrieval_query,
                "current_grade": current_grade,
                "target_grade": target_grade,
                "difficulty_factors": (
                    difficulty_factors or []
                ),
                "factor_training_recommendations": (
                    factor_training_recommendations or []
                ),
                "recommendation": recommendation,
                "result_count": len(retrieved_results),
            },
            sources=sources,
        )

    except ValueError as exc:
        return ToolResult(
            tool="training_recommendation",
            success=False,
            summary=(
                "A training recommendation could not "
                "be generated."
            ),
            data={
                "question": question,
                "current_grade": current_grade,
                "target_grade": target_grade,
                "difficulty_factors": difficulty_factors or [],
                "top_k": top_k,
            },
            error=str(exc),
        )

    except Exception as exc:
        return ToolResult(
            tool="training_recommendation",
            success=False,
            summary=(
                "An unexpected error occurred while "
                "generating a training recommendation."
            ),
            data={
                "question": question,
                "current_grade": current_grade,
                "target_grade": target_grade,
                "difficulty_factors": difficulty_factors or [],
                "top_k": top_k,
            },
            error=str(exc),
        )