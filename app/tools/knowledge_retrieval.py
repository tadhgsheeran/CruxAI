from app.retrieval.service import retrieval_service
from app.tools.schemas import ToolResult, ToolSource


def knowledge_retrieval_tool(
    query: str,
    top_k: int = 3,
) -> ToolResult:
    """
    Retrieve relevant climbing knowledge for a user question.
    """
    try:
        results = retrieval_service.search(
            query=query,
            top_k=top_k,
        )

        sources = [
            ToolSource(
                document=result["source"],
                chunk_id=result["chunk_id"],
                score=result["score"],
                text=result["text"],
            )
            for result in results
        ]

        return ToolResult(
            tool="knowledge_retrieval",
            success=True,
            summary=(
                f"Retrieved {len(results)} relevant "
                "climbing knowledge sources."
            ),
            data={
                "query": query,
                "top_k": top_k,
                "result_count": len(results),
            },
            sources=sources,
        )

    except ValueError as exc:
        return ToolResult(
            tool="knowledge_retrieval",
            success=False,
            summary="Climbing knowledge could not be retrieved.",
            data={
                "query": query,
                "top_k": top_k,
            },
            error=str(exc),
        )

    except Exception as exc:
        return ToolResult(
            tool="knowledge_retrieval",
            success=False,
            summary=(
                "An unexpected error occurred during "
                "knowledge retrieval."
            ),
            data={
                "query": query,
                "top_k": top_k,
            },
            error=str(exc),
        )