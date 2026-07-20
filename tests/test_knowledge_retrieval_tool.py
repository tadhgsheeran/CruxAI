from app.tools.knowledge_retrieval import (
    knowledge_retrieval_tool,
)


def test_knowledge_retrieval_tool_returns_results():
    result = knowledge_retrieval_tool(
        query=(
            "How can I keep my feet on the wall "
            "while climbing an overhang?"
        ),
        top_k=3,
    )

    assert result.success is True
    assert result.tool == "knowledge_retrieval"
    assert result.data["result_count"] == 3
    assert len(result.sources) == 3
    assert result.error is None

    for source in result.sources:
        assert source.document
        assert source.chunk_id
        assert source.score is not None
        assert source.text


def test_knowledge_retrieval_tool_rejects_empty_query():
    result = knowledge_retrieval_tool(
        query="   ",
        top_k=3,
    )

    assert result.success is False
    assert result.tool == "knowledge_retrieval"
    assert result.error is not None
    assert "empty" in result.error.lower()


def test_knowledge_retrieval_tool_rejects_invalid_top_k():
    result = knowledge_retrieval_tool(
        query="How do heel hooks work?",
        top_k=0,
    )

    assert result.success is False
    assert result.error is not None
    assert "greater than 0" in result.error.lower()