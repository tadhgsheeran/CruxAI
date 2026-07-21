PROMPT_VERSIONS = {
    "baseline",
    "evidence_grounded",
}


def get_system_prompt(
    prompt_version: str,
) -> str:
    """
    Return the system prompt for one experiment version.
    """
    if prompt_version == "baseline":
        return (
            "You are CruxAI, a climbing assistant. "
            "Answer the user's climbing question clearly "
            "and concisely."
        )

    if prompt_version == "evidence_grounded":
        return (
            "You are CruxAI, a climbing training assistant. "
            "Answer using only the supplied context. "
            "Do not invent information that is not supported "
            "by the context. Cite factual recommendations "
            "using the source filename in square brackets, "
            "such as [footwork.md]. If the context does not "
            "contain enough relevant information, say: "
            "'I do not have enough relevant information in "
            "the climbing knowledge base to answer that "
            "question.'"
        )

    raise ValueError(
        "prompt_version must be one of: "
        "baseline, evidence_grounded."
    )


def build_user_prompt(
    query: str,
    context: str,
) -> str:
    """
    Build the shared user message.
    """
    return (
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Provide a clear and concise answer."
    )