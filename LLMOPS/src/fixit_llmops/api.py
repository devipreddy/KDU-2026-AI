from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from pydantic import BaseModel

from .analysis import CostAnalyzer
from .service import FixItService
from .storage import SQLiteStateStore
from .models import SupportRequest


class QueryRequest(BaseModel):
    query: str
    customer_id: str | None = None
    metadata: dict[str, str] | None = None


def create_app(config_path: str | Path = "config/app.yaml") -> FastAPI:
    app = FastAPI(title="FixIt AI Support System", version="0.1.0")
    service = FixItService(config_path)
    app.state.service = service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": service.config.app.name}

    @app.post("/query")
    def query(payload: QueryRequest) -> dict:
        response = service.process(
            SupportRequest(
                query=payload.query,
                customer_id=payload.customer_id,
                metadata=payload.metadata or {},
            )
        )
        return response.model_dump()

    @app.post("/admin/reload")
    def reload_config() -> dict[str, str]:
        service.reload(force=True)
        return {"status": "reloaded"}

    @app.get("/admin/stats")
    def stats() -> dict[str, float]:
        analyzer = CostAnalyzer(service.config)
        store = SQLiteStateStore(service.project_root / service.config.app.state_db_path)
        tz = ZoneInfo(service.config.app.timezone)
        now = datetime.now(tz)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if now.month == 12:
            month_end = now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            month_end = now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return {
            "legacy_monthly_cost_usd": analyzer.legacy_monthly_cost(),
            "projected_monthly_cost_usd": analyzer.projected_monthly_cost(),
            "projected_savings_usd": analyzer.savings_amount(),
            "projected_savings_percent": analyzer.savings_percent(),
            "monthly_budget_usd": service.config.budget.monthly_limit_usd,
            "tracked_monthly_spend_usd": store.get_spend_between(
                month_start.astimezone(timezone.utc).isoformat(),
                month_end.astimezone(timezone.utc).isoformat(),
            ),
        }

    return app


app = create_app()
