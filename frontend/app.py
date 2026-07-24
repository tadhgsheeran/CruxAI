import os
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv(
    "CRUXAI_API_URL",
    "http://127.0.0.1:8000",
)

BOARD_ROWS = 18
BOARD_COLUMNS = 11

st.set_page_config(
    page_title="CruxAI",
    page_icon="🧗",
    layout="wide",
)

st.title("CruxAI")
st.subheader("AI Climbing Coach and Route-Analysis System")

st.write(
    "Ask climbing questions, request training advice, "
    "or build a MoonBoard route for full analysis."
)


def create_empty_route() -> list[list[int]]:
    """
    Create an empty 18-by-11 MoonBoard route.
    """
    return [
        [0 for _ in range(BOARD_COLUMNS)]
        for _ in range(BOARD_ROWS)
    ]


def initialize_route_state() -> None:
    """
    Initialize the interactive board in Streamlit session state.
    """
    if "moonboard_route" not in st.session_state:
        st.session_state.moonboard_route = (
            create_empty_route()
        )


def reset_route() -> None:
    """
    Clear every selected hold.
    """
    st.session_state.moonboard_route = (
        create_empty_route()
    )

    for row in range(BOARD_ROWS):
        for column in range(BOARD_COLUMNS):
            key = f"hold_{row}_{column}"

            if key in st.session_state:
                st.session_state[key] = False


def build_route_from_widgets() -> list[list[int]]:
    """
    Convert the board checkboxes into an 18-by-11 binary grid.
    """
    route = create_empty_route()

    for row in range(BOARD_ROWS):
        for column in range(BOARD_COLUMNS):
            key = f"hold_{row}_{column}"

            route[row][column] = int(
                st.session_state.get(
                    key,
                    False,
                )
            )

    st.session_state.moonboard_route = route

    return route


def count_active_holds(
    route: list[list[int]],
) -> int:
    """
    Count selected holds on the board.
    """
    return sum(
        sum(row)
        for row in route
    )


def call_analyze_endpoint(
    question: str,
    current_grade: int | None,
    target_grade: int | None,
    top_k: int,
    route: list[list[int]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": question,
        "top_k": top_k,
    }

    if current_grade is not None:
        payload["current_grade"] = current_grade

    if target_grade is not None:
        payload["target_grade"] = target_grade

    if route is not None:
        payload["route"] = route

    response = requests.post(
        f"{API_BASE_URL}/analyze",
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


initialize_route_state()


with st.sidebar:
    st.header("Analysis Settings")

    include_current_grade = st.checkbox(
        "Include current grade",
        value=True,
    )

    current_grade = None

    if include_current_grade:
        current_grade = st.number_input(
            "Current V grade",
            min_value=0,
            max_value=17,
            value=5,
            step=1,
        )

    include_target_grade = st.checkbox(
        "Include target grade",
        value=True,
    )

    target_grade = None

    if include_target_grade:
        target_grade = st.number_input(
            "Target V grade",
            min_value=0,
            max_value=17,
            value=7,
            step=1,
        )

    top_k = st.slider(
        "Retrieved sources",
        min_value=1,
        max_value=5,
        value=3,
    )

    include_route = st.checkbox(
        "Include MoonBoard route",
        value=False,
    )


question = st.text_area(
    "What would you like help with?",
    value=(
        "How hard is this route, what makes it difficult, "
        "and what should I train?"
        if include_route
        else "What should I train to improve on steep routes?"
    ),
    height=120,
)


if include_route:
    st.header("MoonBoard Route")

    st.write(
        "Select each active hold. Row 18 is shown at the top "
        "and row 1 at the bottom."
    )

    header_columns = st.columns(
        [0.7] + [1] * BOARD_COLUMNS
    )

    header_columns[0].markdown("**Row**")

    for column in range(BOARD_COLUMNS):
        column_label = chr(
            ord("A") + column
        )

        header_columns[column + 1].markdown(
            f"**{column_label}**"
        )

    # Display row 18 at the top and row 1 at the bottom.
    for display_row in reversed(
        range(BOARD_ROWS)
    ):
        grid_columns = st.columns(
            [0.7] + [1] * BOARD_COLUMNS
        )

        grid_columns[0].markdown(
            f"**{display_row + 1}**"
        )

        for column in range(BOARD_COLUMNS):
            key = (
                f"hold_{display_row}_{column}"
            )

            grid_columns[column + 1].checkbox(
                label=(
                    f"Row {display_row + 1}, "
                    f"Column {column + 1}"
                ),
                key=key,
                label_visibility="collapsed",
            )

    selected_route = build_route_from_widgets()

    active_hold_count = count_active_holds(
        selected_route
    )

    board_metric_columns = st.columns(2)

    board_metric_columns[0].metric(
        "Selected holds",
        active_hold_count,
    )

    board_metric_columns[1].metric(
        "Board shape",
        "18 × 11",
    )

    if st.button(
        "Clear Route",
        use_container_width=True,
    ):
        reset_route()
        st.rerun()

    with st.expander("View route matrix"):
        st.json(selected_route)

else:
    selected_route = None
    active_hold_count = 0


analyze_clicked = st.button(
    "Analyze",
    type="primary",
    use_container_width=True,
)


if analyze_clicked:
    if not question.strip():
        st.error(
            "Enter a climbing question."
        )

    elif (
        include_route
        and active_hold_count == 0
    ):
        st.error(
            "Select at least one hold before "
            "analyzing a route."
        )

    else:
        with st.spinner(
            "CruxAI is analyzing your request..."
        ):
            try:
                result = call_analyze_endpoint(
                    question=question,
                    current_grade=(
                        int(current_grade)
                        if current_grade is not None
                        else None
                    ),
                    target_grade=(
                        int(target_grade)
                        if target_grade is not None
                        else None
                    ),
                    top_k=top_k,
                    route=selected_route,
                )

            except requests.ConnectionError:
                st.error(
                    "Could not connect to the CruxAI API. "
                    "Confirm that FastAPI is running."
                )
                st.stop()

            except requests.Timeout:
                st.error(
                    "The request timed out before "
                    "the API responded."
                )
                st.stop()

            except requests.HTTPError as exc:
                st.error(
                    f"The API returned an error: {exc}"
                )

                if exc.response is not None:
                    try:
                        st.json(
                            exc.response.json()
                        )
                    except ValueError:
                        st.write(
                            exc.response.text
                        )

                st.stop()

            except requests.RequestException as exc:
                st.error(
                    f"Request failed: {exc}"
                )
                st.stop()

        if result.get("success"):
            st.success("Analysis complete")
        else:
            st.warning(
                "CruxAI could not complete every "
                "requested step."
            )

        final_answer = result.get(
            "final_answer"
        )

        if final_answer:
            st.markdown(final_answer)
        else:
            st.info(
                "No final answer was returned."
            )

        metadata = result.get(
            "metadata",
            {},
        )

        metric_columns = st.columns(4)

        metric_columns[0].metric(
            "Intent",
            result.get(
                "intent",
                "Unknown",
            ),
        )

        metric_columns[1].metric(
            "Tools used",
            len(
                result.get(
                    "selected_tools",
                    [],
                )
            ),
        )

        total_latency = metadata.get(
            "total_latency_seconds",
            0,
        )

        metric_columns[2].metric(
            "Total latency",
            f"{total_latency:.2f}s",
        )
        metric_columns[3].metric(
            "Tools succeeded",
            metadata.get(
                "tools_succeeded",
                0,
            ),
        )

        with st.expander(
            "Routing and workflow details"
        ):
            st.write(
                "**Selected tools:**",
                result.get(
                    "selected_tools",
                    [],
                ),
            )

            st.write(
                "**Routing reason:**",
                metadata.get(
                    "routing_reason",
                    "Not available",
                ),
            )

            st.json(metadata)

        tool_results = result.get(
            "tool_results",
            {},
        )

        if tool_results:
            st.header("Tool Results")

            for (
                tool_name,
                tool_result,
            ) in tool_results.items():
                with st.expander(
                    tool_name.replace(
                        "_",
                        " ",
                    ).title()
                ):
                    st.write(
                        tool_result.get(
                            "summary",
                            "No summary available.",
                        )
                    )

                    tool_data = tool_result.get(
                        "data",
                        {},
                    )

                    if (
                        tool_name
                        == "grade_prediction"
                    ):
                        formatted_grade = (
                            tool_data.get(
                                "formatted_grade"
                            )
                        )

                        predicted_grade = (
                            tool_data.get(
                                "predicted_grade"
                            )
                        )

                        grade_columns = (
                            st.columns(2)
                        )

                        grade_columns[0].metric(
                            "Predicted grade",
                            formatted_grade
                            or "Unknown",
                        )

                        if (
                            predicted_grade
                            is not None
                        ):
                            grade_columns[1].metric(
                                "Raw prediction",
                                f"{predicted_grade:.2f}",
                            )

                    if (
                        tool_name
                        == "difficulty_analysis"
                    ):
                        factors = tool_data.get(
                            "difficulty_factors",
                            [],
                        )

                        if factors:
                            st.subheader(
                                "Difficulty Factors"
                            )

                            for factor in factors:
                                st.markdown(
                                    f"- {factor}"
                                )

                    sources = tool_result.get(
                        "sources",
                        [],
                    )

                    if sources:
                        st.subheader("Sources")

                        for source in sources:
                            document = source.get(
                                "document",
                                "Unknown source",
                            )

                            score = source.get(
                                "score"
                            )

                            if score is not None:
                                st.markdown(
                                    f"**{document}** — "
                                    f"score: `{score:.3f}`"
                                )
                            else:
                                st.markdown(
                                    f"**{document}**"
                                )

                            source_text = (
                                source.get("text")
                            )

                            if source_text:
                                st.write(
                                    source_text
                                )

                            st.divider()

                    with st.expander(
                        "Raw tool data"
                    ):
                        st.json(tool_data)

        errors = result.get(
            "errors",
            [],
        )

        if errors:
            st.header("Errors")

            for error in errors:
                st.error(error)