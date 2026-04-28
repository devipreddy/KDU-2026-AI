from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_search_service
from app.api.security import require_api_key
from app.db.session import get_db
from app.schemas.search import SearchRequest, SearchResponse
from app.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["search"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=SearchResponse)
def search_content(
    payload: SearchRequest,
    db: Session = Depends(get_db),
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    try:
        return service.search(db=db, query=payload.query, file_id=payload.file_id, top_k=payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
