from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class HealthResponse(APIModel):
    status: str
    version: str
    timestamp: datetime
