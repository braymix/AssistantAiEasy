"""Query endpoint – combines knowledge retrieval with LLM generation."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.gateway.schemas.query import QueryRequest, QueryResponse
from src.knowledge.service import KnowledgeService
from src.llm.base import get_llm_provider
from src.shared.database import get_db_session

router = APIRouter()


@router.post("/ask", response_model=QueryResponse)
async def ask(
    payload: QueryRequest,
    session: AsyncSession = Depends(get_db_session),
) -> QueryResponse:
    """Retrieve relevant knowledge and generate an LLM-powered answer."""
    service = KnowledgeService(session)

    # 1. Retrieve relevant chunks from the vector store
    chunks = await service.search(payload.question, top_k=payload.top_k)

    # 2. Build context from retrieved chunks
    context = "\n\n---\n\n".join(chunk.content for chunk in chunks)

    # 3. Generate answer via LLM
    llm = get_llm_provider()
    prompt = (
        f"Based on the following context, answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {payload.question}\n\n"
        f"Answer:"
    )
    answer = await llm.generate(prompt)

    return QueryResponse(
        answer=answer,
        sources=[
            {"title": c.title, "score": c.score}
            for c in chunks
        ],
    )
