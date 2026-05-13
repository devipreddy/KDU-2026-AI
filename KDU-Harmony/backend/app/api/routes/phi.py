from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.phi import PhiLookupRequest, PhiLookupResponse
from app.services.phi_store import (
    PhiLookupDeniedError,
    PhiMappingNotFoundError,
    lookup_phi_value_for_user,
)

router = APIRouter(prefix="/phi")
DbSession = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]


@router.post("/lookup", response_model=PhiLookupResponse)
def lookup_phi_value(
    payload: PhiLookupRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> PhiLookupResponse:
    try:
        result = lookup_phi_value_for_user(
            db,
            token=payload.token,
            patient_ref=payload.patient_ref,
            current_user=current_user,
        )
    except PhiMappingNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PHI token was not found",
        ) from exc
    except PhiLookupDeniedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="PHI lookup is not permitted for this user",
        ) from exc

    return PhiLookupResponse(
        mapping_id=result.mapping_id,
        token=result.token,
        patient_ref=result.patient_ref,
        entity_type=result.entity_type,
        value=result.value,
        decrypted_at=result.decrypted_at,
    )
