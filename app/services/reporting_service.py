from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.database import ApiUsageRecord, FileRecord
from app.schemas.cost import CostDashboardResponse, CostModelSummary, CostOperationSummary, FileCostSummary
from app.services.costs import classify_operation


class ReportingService:
    def build_cost_dashboard(self, db: Session) -> CostDashboardResponse:
        usage_records = db.scalars(select(ApiUsageRecord).order_by(ApiUsageRecord.created_at.desc())).all()
        file_map = {file.id: file.file_name for file in db.scalars(select(FileRecord)).all()}

        by_operation: dict[str, dict[str, float | int]] = defaultdict(lambda: self._blank_bucket())
        by_model: dict[str, dict[str, float | int]] = defaultdict(lambda: self._blank_bucket())
        by_file: dict[str, dict[str, float | int | str]] = defaultdict(
            lambda: {
                "file_name": "",
                "vision_cost_usd": 0.0,
                "enrichment_cost_usd": 0.0,
                "embedding_cost_usd": 0.0,
                "total_cost_usd": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }
        )

        total_cost = 0.0
        for record in usage_records:
            total_cost += record.estimated_cost_usd
            self._accumulate(by_operation[record.operation], record)
            self._accumulate(by_model[record.model], record)

            if record.file_id:
                file_bucket = by_file[record.file_id]
                file_bucket["file_name"] = file_map.get(record.file_id, "Unknown")
                file_bucket["total_cost_usd"] += record.estimated_cost_usd
                file_bucket["total_input_tokens"] += record.input_tokens
                file_bucket["total_output_tokens"] += record.output_tokens

                category = classify_operation(record.operation)
                if category == "vision":
                    file_bucket["vision_cost_usd"] += record.estimated_cost_usd
                elif category == "enrichment":
                    file_bucket["enrichment_cost_usd"] += record.estimated_cost_usd
                elif category == "embedding":
                    file_bucket["embedding_cost_usd"] += record.estimated_cost_usd

        files = [
            FileCostSummary(
                file_id=file_id,
                file_name=str(data["file_name"]),
                vision_cost_usd=round(float(data["vision_cost_usd"]), 6),
                enrichment_cost_usd=round(float(data["enrichment_cost_usd"]), 6),
                embedding_cost_usd=round(float(data["embedding_cost_usd"]), 6),
                total_cost_usd=round(float(data["total_cost_usd"]), 6),
                total_input_tokens=int(data["total_input_tokens"]),
                total_output_tokens=int(data["total_output_tokens"]),
            )
            for file_id, data in by_file.items()
        ]
        files.sort(key=lambda item: item.total_cost_usd, reverse=True)

        operations = [
            CostOperationSummary(
                operation=operation,
                input_tokens=int(values["input_tokens"]),
                output_tokens=int(values["output_tokens"]),
                total_tokens=int(values["total_tokens"]),
                estimated_cost_usd=round(float(values["estimated_cost_usd"]), 6),
            )
            for operation, values in by_operation.items()
        ]
        operations.sort(key=lambda item: item.estimated_cost_usd, reverse=True)

        models = [
            CostModelSummary(
                model=model,
                input_tokens=int(values["input_tokens"]),
                output_tokens=int(values["output_tokens"]),
                total_tokens=int(values["total_tokens"]),
                estimated_cost_usd=round(float(values["estimated_cost_usd"]), 6),
            )
            for model, values in by_model.items()
        ]
        models.sort(key=lambda item: item.estimated_cost_usd, reverse=True)

        return CostDashboardResponse(
            total_cost_usd=round(total_cost, 6),
            files=files,
            by_operation=operations,
            by_model=models,
        )

    def _accumulate(self, bucket: dict[str, float | int], record: ApiUsageRecord) -> None:
        bucket["input_tokens"] += record.input_tokens
        bucket["output_tokens"] += record.output_tokens
        bucket["total_tokens"] += record.total_tokens
        bucket["estimated_cost_usd"] += record.estimated_cost_usd

    def _blank_bucket(self) -> dict[str, float | int]:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
