from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH
from .schemas import SummaryLength
from .workflow import build_assistant


def main() -> int:
    args = _build_parser().parse_args()
    _configure_logging(args.verbose)

    assistant = build_assistant(args.config)
    source_text = _load_input_text(args.input_file)
    summary_length = SummaryLength.from_value(args.length or _prompt_for_length())

    artifacts = assistant.generate_summary(source_text, summary_length)

    print("\n=== Final Summary ===")
    print(textwrap.fill(artifacts.final_summary, width=100))
    print(f"\nLength profile: {artifacts.length.value}")
    print(f"Initial chunk count: {artifacts.chunk_count}")

    if args.no_interactive:
        return 0

    print("\nAsk questions about the summary. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if question.lower() in {"exit", "quit"}:
            print("Exiting.")
            return 0

        if not question:
            continue

        answer = assistant.answer_question(question, artifacts.final_summary)
        print(f"Answer: {answer.answer}")
        print(f"Confidence: {answer.score:.3f}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a locally hosted tri-model assistant for summarization and question answering."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--input-file",
        help="Path to a text file to summarize. If omitted, paste text directly into the terminal.",
    )
    parser.add_argument(
        "--length",
        choices=[item.value for item in SummaryLength],
        help="Target summary length.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Generate the summary and exit without entering the QA loop.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _load_input_text(input_file: str | None) -> str:
    if input_file:
        return Path(input_file).read_text(encoding="utf-8")

    print("Paste the source text below. Enter a line containing only END to finish.\n")
    lines: list[str] = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def _prompt_for_length() -> str:
    options = ", ".join(item.value for item in SummaryLength)
    while True:
        choice = input(f"Choose summary length ({options}): ").strip().lower()
        if choice in {item.value for item in SummaryLength}:
            return choice
        print("Please enter short, medium, or long.")
