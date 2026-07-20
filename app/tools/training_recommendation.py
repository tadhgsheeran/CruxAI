from app.generation.service import generation_service
from app.retrieval.service import retrieval_service
from app.tools.schemas import ToolResult, ToolSource


def build_training_query(
    question: str,
    current_grade: int | None = None,
    target_grade: int | None = None,
) -> str:
    """
    Add optional climber context to the retrieval query.
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

    return " ".join(query_parts)


def training_recommendation_tool(
    question: str,
    current_grade: int | None = None,
    target_grade: int | None = None,
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
        )

        retrieved_results = retrieval_service.search(
            query=retrieval_query,
            top_k=top_k,
        )

        recommendation = generation_service.generate_answer(
            query=retrieval_query,
            retrieved_results=retrieved_results,
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
                "top_k": top_k,
            },
            error=str(exc),
        )