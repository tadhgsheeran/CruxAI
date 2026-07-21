from app.tools.schemas import ToolResult


def synthesize_response(
    question: str,
    tool_results: dict[str, ToolResult],
) -> str:
    """
    Combine successful tool results into one user-facing answer.

    This first version is deterministic. It does not make another
    language-model call.
    """
    successful_results = {
        name: result
        for name, result in tool_results.items()
        if result.success
    }

    if not successful_results:
        return (
            "CruxAI could not complete the requested analysis."
        )

    answer_sections = []

    grade_result = successful_results.get(
        "grade_prediction"
    )

    if grade_result is not None:
        formatted_grade = grade_result.data.get(
            "formatted_grade"
        )

        predicted_grade = grade_result.data.get(
            "predicted_grade"
        )

        if formatted_grade:
            grade_text = (
                f"The route is predicted to be "
                f"{formatted_grade}"
            )

            if predicted_grade is not None:
                grade_text += (
                    f" with a raw model prediction of "
                    f"{predicted_grade:.2f}"
                )

            grade_text += "."

            answer_sections.append(grade_text)

    difficulty_result = successful_results.get(
        "difficulty_analysis"
    )

    if difficulty_result is not None:
        factors = difficulty_result.data.get(
            "difficulty_factors",
            [],
        )
    
        hold_count = difficulty_result.data.get(
            "hold_count"
        )
    
        vertical_span = difficulty_result.data.get(
            "vertical_span"
        )
    
        horizontal_span = difficulty_result.data.get(
            "horizontal_span"
        )

        average_move_distance = (
            difficulty_result.data.get(
                "average_move_distance"
            )
        )

        difficulty_lines = []

        if hold_count is not None:
            difficulty_lines.append(
                f"- Active holds: {hold_count}"
            )

        if vertical_span is not None:
            difficulty_lines.append(
                f"- Vertical span: {vertical_span} rows"
            )

        if horizontal_span is not None:
            difficulty_lines.append(
                f"- Horizontal span: {horizontal_span} columns"
            )

        if average_move_distance is not None:
            difficulty_lines.append(
                "- Estimated average move distance: "
                f"{average_move_distance:.2f} grid units"
            )

        for factor in factors:
            difficulty_lines.append(
                f"- {factor}"
            )

        if difficulty_lines:
            answer_sections.append(
                "Route difficulty factors:\n"
                + "\n".join(difficulty_lines)
            )
            
    similar_result = successful_results.get(
        "similar_route_search"
    )

    if similar_result is not None:
        matches = similar_result.data.get(
            "matches",
            [],
        )

        if matches:
            match_lines = []

            for match in matches[:5]:
                match_lines.append(
                    "- Route "
                    f"{match['route_id']}: "
                    f"{match['formatted_grade']} "
                    f"(similarity "
                    f"{match['similarity']:.3f})"
                )

            answer_sections.append(
                "The closest matching routes are:\n"
                + "\n".join(match_lines)
            )
        else:
            answer_sections.append(
                "No similar routes were returned."
            )

    training_result = successful_results.get(
        "training_recommendation"
    )

    if training_result is not None:
        factor_training = training_result.data.get(
            "factor_training_recommendations",
            [],
        )

        if factor_training:
            training_lines = []

            for item in factor_training:
                focus = item.get(
                    "focus",
                    "training focus",
                )

                reason = item.get(
                    "reason",
                    "",
                )

                methods = item.get(
                    "methods",
                    [],
                )

                training_lines.append(
                    f"- {focus.title()}: {reason}"
                )

                for method in methods:
                    training_lines.append(
                        f"  - {method.capitalize()}"
                    )

            answer_sections.append(
                "Route-specific training priorities:\n"
                + "\n".join(training_lines)
            )

        else:
            recommendation = training_result.data.get(
                "recommendation"
            )

            if recommendation:
                answer_sections.append(
                    recommendation
                )

    retrieval_result = successful_results.get(
        "knowledge_retrieval"
    )

    if retrieval_result is not None:
        sources = retrieval_result.sources

        if sources:
            evidence_sections = []

            for source in sources:
                if source.text:
                    evidence_sections.append(
                        source.text.strip()
                    )

            if evidence_sections:
                answer_sections.append(
                    "\n\n".join(evidence_sections)
                )

            source_names = list(
                dict.fromkeys(
                    source.document
                    for source in sources
                )
            )

            if source_names:
                citations = " ".join(
                    f"[{source_name}]"
                    for source_name in source_names
                )

                answer_sections.append(
                    f"Sources: {citations}"
                )

    if not answer_sections:
        summaries = [
            result.summary
            for result in successful_results.values()
        ]

        answer_sections.extend(summaries)

    return "\n\n".join(answer_sections)