from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from .analysis import CostAnalyzer
from .models import SupportRequest
from .service import FixItService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FixIt local LLMOps service.")
    parser.add_argument("--config", default="config/app.yaml", help="Path to the YAML configuration file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the FastAPI server.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    query = subparsers.add_parser("query", help="Run a single support query through the router.")
    query.add_argument("text", help="Customer query text.")

    subparsers.add_parser("cost-report", help="Print projected cost savings.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config)

    if args.command == "serve":
        uvicorn.run("fixit_llmops.api:app", host=args.host, port=args.port, reload=False)
        return 0

    service = FixItService(config_path)
    if args.command == "query":
        response = service.process(SupportRequest(query=args.text))
        print(json.dumps(response.model_dump(), indent=2))
        return 0

    analyzer = CostAnalyzer(service.config)
    report = {
        "legacy_monthly_cost_usd": analyzer.legacy_monthly_cost(),
        "projected_monthly_cost_usd": analyzer.projected_monthly_cost(),
        "savings_amount_usd": analyzer.savings_amount(),
        "savings_percent": analyzer.savings_percent(),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

