from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_reporting_service
from app.api.security import require_api_key
from app.db.session import get_db
from app.schemas.cost import CostDashboardResponse
from app.services.reporting_service import ReportingService

router = APIRouter(prefix="/costs", tags=["costs"], dependencies=[Depends(require_api_key)])


@router.get("", response_model=CostDashboardResponse)
def get_cost_dashboard(
    db: Session = Depends(get_db),
    service: ReportingService = Depends(get_reporting_service),
) -> CostDashboardResponse:
    return service.build_cost_dashboard(db)
