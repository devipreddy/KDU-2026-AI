from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, role_names_for
from app.core.observability import langsmith_trace
from app.db.session import get_db
from app.models.user import User
from app.schemas.search import (
    SearchAnswerResponse,
    SearchHitResponse,
    SearchRequest,
    SearchResponse,
    SearchUserResponse,
)
from app.services.cross_encoder_reranking import RerankingError
from app.services.embedding_pipeline import EmbeddingPipelineError
from app.services.llm_answer import (
    LLMAnswerResult,
    generate_evidence_grounded_answer,
    stream_evidence_grounded_answer,
    streaming_placeholder_answer,
)
from app.services.result_rendering import PHIRenderedSearchResult, phi_aware_search

router = APIRouter(prefix="/search")

DbSession = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]


@router.post("", response_model=SearchResponse)
def run_search(
    payload: SearchRequest,
    request: Request,
    current_user: CurrentUser,
    db: DbSession,
) -> SearchResponse:
    trace_inputs = {
        "query": payload.query,
        "user_id": str(current_user.id),
        "roles": role_names_for(current_user),
        "limit": payload.limit,
        "candidate_limit": payload.candidate_limit,
        "rerank_top_n": payload.rerank_top_n,
        "include_llm_answer": payload.include_llm_answer,
    }
    with langsmith_trace("api.search_pipeline", inputs=trace_inputs) as trace_payload:
        try:
            rendered_result = phi_aware_search(
                db,
                user=current_user,
                query=payload.query,
                limit=payload.limit,
                candidate_limit=payload.candidate_limit,
                rerank_top_n=payload.rerank_top_n,
                collection_name=payload.collection,
                audit=True,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )
        except (EmbeddingPipelineError, RerankingError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

        llm_answer = generate_evidence_grounded_answer(
            rendered_result,
            enabled=payload.include_llm_answer,
        )
        trace_payload["outputs"] = {
            "hit_count": len(rendered_result.hits),
            "llm_status": llm_answer.status,
            "render_mode": rendered_result.rendering_policy.render_mode,
            "access_denied": rendered_result.authorization.denied,
        }
        if payload.require_llm_answer and llm_answer.status != "generated":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "message": "LLM answer generation was required but did not complete.",
                    "llm_status": llm_answer.status,
                    "llm_error": llm_answer.error,
                },
            )

    return serialize_search_response(
        rendered_result=rendered_result,
        llm_answer=llm_answer,
        user=current_user,
    )


@router.post("/stream")
def stream_search(
    payload: SearchRequest,
    request: Request,
    current_user: CurrentUser,
    db: DbSession,
) -> StreamingResponse:
    trace_inputs = {
        "query": payload.query,
        "user_id": str(current_user.id),
        "roles": role_names_for(current_user),
        "limit": payload.limit,
        "candidate_limit": payload.candidate_limit,
        "rerank_top_n": payload.rerank_top_n,
        "include_llm_answer": payload.include_llm_answer,
        "streaming": True,
    }
    with langsmith_trace("api.search_pipeline.stream", inputs=trace_inputs) as trace_payload:
        try:
            rendered_result = phi_aware_search(
                db,
                user=current_user,
                query=payload.query,
                limit=payload.limit,
                candidate_limit=payload.candidate_limit,
                rerank_top_n=payload.rerank_top_n,
                collection_name=payload.collection,
                audit=True,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )
        except (EmbeddingPipelineError, RerankingError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

        trace_payload["outputs"] = {
            "hit_count": len(rendered_result.hits),
            "render_mode": rendered_result.rendering_policy.render_mode,
            "access_denied": rendered_result.authorization.denied,
            "streaming": True,
        }

    placeholder_answer = streaming_placeholder_answer(
        rendered_result,
        enabled=payload.include_llm_answer,
    )
    retrieval_response = serialize_search_response(
        rendered_result=rendered_result,
        llm_answer=placeholder_answer,
        user=current_user,
    )

    def event_stream():
        yield sse_event("retrieval", retrieval_response)
        final_answer = placeholder_answer
        for answer_event in stream_evidence_grounded_answer(
            rendered_result,
            enabled=payload.include_llm_answer,
        ):
            if answer_event.event == "delta" and answer_event.delta:
                yield sse_event("answer_delta", {"delta": answer_event.delta})
                continue
            if answer_event.event == "done" and answer_event.result is not None:
                final_answer = answer_event.result
                if payload.require_llm_answer and final_answer.status != "generated":
                    yield sse_event(
                        "error",
                        {
                            "message": "LLM answer generation was required but did not complete.",
                            "llm_status": final_answer.status,
                            "llm_error": final_answer.error,
                        },
                    )
                    return
                final_response = serialize_search_response(
                    rendered_result=rendered_result,
                    llm_answer=final_answer,
                    user=current_user,
                )
                yield sse_event("answer_done", final_response)
                return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def sse_event(event: str, data: Any) -> str:
    payload = json.dumps(jsonable_encoder(data), separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


def serialize_search_response(
    *,
    rendered_result: PHIRenderedSearchResult,
    llm_answer: LLMAnswerResult,
    user: User,
) -> SearchResponse:
    result_metadata = rendered_result.to_metadata()
    return SearchResponse(
        query=rendered_result.query,
        user=SearchUserResponse(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            roles=role_names_for(user),
        ),
        hit_count=len(rendered_result.hits),
        answer=SearchAnswerResponse(
            status=llm_answer.status,
            answer=llm_answer.answer,
            provider=llm_answer.provider,
            model=llm_answer.model,
            citations=[citation.to_metadata() for citation in llm_answer.citations],
            latency_ms=llm_answer.latency_ms,
            error=llm_answer.error,
        ),
        hits=[SearchHitResponse(**hit.to_metadata()) for hit in rendered_result.hits],
        timeline=result_metadata["timeline"],
        pipeline=pipeline_metadata(rendered_result, llm_answer),
    )


def pipeline_metadata(
    rendered_result: PHIRenderedSearchResult,
    llm_answer: LLMAnswerResult,
) -> dict[str, Any]:
    context_result = rendered_result.context_result
    rerank_result = context_result.rerank_result
    hybrid_result = rerank_result.hybrid_result
    bm25_result = hybrid_result.bm25_result
    dense_result = hybrid_result.dense_result
    llm_metadata = llm_answer.to_metadata()
    llm_metadata.pop("answer", None)
    return {
        "query_understanding": {
            "metadata_filters": rendered_result.authorization.query_metadata_filters,
        },
        "authorization": rendered_result.authorization.to_metadata(),
        "bm25": {
            "retriever": "local_bm25_retriever_v1",
            "candidate_count": bm25_result.candidate_chunk_count,
            "hit_count": len(bm25_result.hits),
            "query_terms": bm25_result.query_terms.to_metadata(),
        },
        "dense": {
            "retriever": "chroma_dense_retriever_v1",
            "collection_name": dense_result.collection_name,
            "embedding_model": dense_result.query_embedding_model,
            "embedding_dimension": dense_result.query_embedding_dimension,
            "candidate_count": dense_result.candidate_count,
            "hit_count": len(dense_result.hits),
        },
        "fusion": {
            "retriever": "rrf_hybrid_retriever_v1",
            "algorithm": "reciprocal_rank_fusion",
            "rrf_k": hybrid_result.rrf_k,
            "candidate_limit": hybrid_result.candidate_limit,
            "hit_count": len(hybrid_result.hits),
        },
        "reranker": {
            "retriever": "cross_encoder_reranker_v1",
            "model": rerank_result.reranker_model,
            "rerank_top_n": rerank_result.rerank_top_n,
            "reranked_count": rerank_result.reranked_count,
        },
        "context_expansion": {
            "strategy": "parent_context_expansion_v1",
            "hit_count": len(context_result.hits),
        },
        "rendering": rendered_result.rendering_policy.to_metadata(),
        "llm": llm_metadata,
    }
