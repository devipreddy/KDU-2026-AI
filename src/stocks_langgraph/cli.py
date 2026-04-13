from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .agent import TradingAgent
from .evaluation import run_evaluation_suite


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stateful LangGraph stock trading agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    message = subparsers.add_parser("message", help="Send a message into a trading thread")
    message.add_argument("--thread-id", required=True)
    message.add_argument("--content", required=True)
    message.add_argument("--user-id")
    message.add_argument("--base-currency", default="INR")

    approve = subparsers.add_parser("approve", help="Approve or reject a pending trade")
    approve.add_argument("--thread-id", required=True)
    approve.add_argument("--approved", required=True, choices=["true", "false"])
    approve.add_argument("--reviewer", default="human-reviewer")
    approve.add_argument("--reason")

    state = subparsers.add_parser("state", help="Inspect the latest thread state")
    state.add_argument("--thread-id", required=True)

    analytics = subparsers.add_parser("analytics", help="Inspect the latest thread analytics")
    analytics.add_argument("--thread-id", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Run the built-in evaluation suite")
    evaluate.add_argument("--output", type=Path)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    agent = TradingAgent.from_env()
    try:
        if args.command == "message":
            result = agent.run_turn(
                thread_id=args.thread_id,
                content=args.content,
                user_id=args.user_id,
                base_currency=args.base_currency,
            )
            print(json.dumps(result.model_dump(mode="json"), indent=2, default=_json_default))
        elif args.command == "approve":
            result = agent.approve(
                thread_id=args.thread_id,
                approved=args.approved == "true",
                reviewer=args.reviewer,
                reason=args.reason,
            )
            print(json.dumps(result.model_dump(mode="json"), indent=2, default=_json_default))
        elif args.command == "state":
            print(json.dumps(agent.get_state(args.thread_id), indent=2, default=_json_default))
        elif args.command == "analytics":
            state = agent.get_state(args.thread_id)
            analytics = agent._build_analytics(state)
            print(json.dumps(analytics.model_dump(mode="json") if analytics else {}, indent=2, default=_json_default))
        elif args.command == "evaluate":
            report = run_evaluation_suite(agent, args.output)
            print(json.dumps(report, indent=2, default=_json_default))
    finally:
        agent.close()


if __name__ == "__main__":
    main()
