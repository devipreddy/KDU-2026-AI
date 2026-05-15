from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class PhiLookupRequest(BaseModel):
    token: str = Field(min_length=3, max_length=160)
    patient_ref: str | None = Field(default=None, max_length=80)


class PhiLookupResponse(BaseModel):
    mapping_id: UUID
    token: str
    patient_ref: str
    entity_type: str
    value: str
    decrypted_at: datetime
