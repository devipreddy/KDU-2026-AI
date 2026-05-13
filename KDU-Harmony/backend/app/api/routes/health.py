from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str


@router.get("/health", response_model=HealthResponse)
def read_api_health() -> HealthResponse:
    return HealthResponse(status="ok", service="healthcare-semantic-search-api")
