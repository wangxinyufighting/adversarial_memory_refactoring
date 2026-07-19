#!/usr/bin/env python3
"""Print paper metrics and per-question accuracy from a comparison JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


PRIMARY_METRICS = ("R@5", "R@10", "N@5", "N@10", "answer_accuracy")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the primary retrieval and answer metrics for the method "
            "and baseline in a memory-QA comparison JSON file."
        )
    )
    parser.add_argument("input", type=Path, help="comparison JSON file")
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="decimal places used for percentages (default: 2)",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {path}: {error}") from error

    if not isinstance(data, dict):
        raise ValueError("the top-level JSON value must be an object")
    return data


def require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"missing or invalid object at {path}")
    return value


def require_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing or invalid number at {path}")
    return float(value)


def method_name(data: dict[str, Any]) -> str:
    comparison = data.get("comparison")
    if isinstance(comparison, dict) and isinstance(comparison.get("method"), str):
        return comparison["method"]
    return "ours"


def baseline_name(data: dict[str, Any]) -> str:
    baseline = data.get("baseline")
    if isinstance(baseline, dict) and isinstance(baseline.get("name"), str):
        return baseline["name"]
    return "baseline"


def format_percent(value: float, digits: int) -> str:
    return f"{value * 100:.{digits}f}%"


def format_delta(value: float, digits: int) -> str:
    return f"{value * 100:+.{digits}f} pp"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        if len(row) != len(headers):
            raise ValueError("table row has a different length from its header")
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render(row: Sequence[str]) -> str:
        return "| " + " | ".join(
            cell.ljust(width) for cell, width in zip(row, widths)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


def build_tables(data: dict[str, Any], digits: int) -> tuple[str, str]:
    if digits < 0:
        raise ValueError("--digits must be zero or greater")

    ours_summary = require_mapping(data.get("summary"), "summary")
    baseline = require_mapping(data.get("baseline"), "baseline")
    baseline_summary = require_mapping(baseline.get("summary"), "baseline.summary")
    ours_metrics = require_mapping(
        ours_summary.get("paper_metrics"), "summary.paper_metrics"
    )
    baseline_metrics = require_mapping(
        baseline_summary.get("paper_metrics"), "baseline.summary.paper_metrics"
    )

    first_rows: list[list[str]] = []
    for metric in PRIMARY_METRICS:
        ours = require_number(
            ours_metrics.get(metric), f"summary.paper_metrics.{metric}"
        )
        base = require_number(
            baseline_metrics.get(metric), f"baseline.summary.paper_metrics.{metric}"
        )
        first_rows.append(
            [
                metric.replace("answer_accuracy", "Answer accuracy"),
                format_percent(ours, digits),
                format_percent(base, digits),
                format_delta(ours - base, digits),
            ]
        )

    ours_by_type = require_mapping(
        ours_summary.get("by_question_type"), "summary.by_question_type"
    )
    baseline_by_type = require_mapping(
        baseline_summary.get("by_question_type"),
        "baseline.summary.by_question_type",
    )
    question_types = sorted(set(ours_by_type) | set(baseline_by_type))
    second_rows: list[list[str]] = []
    for question_type in question_types:
        ours_task = require_mapping(
            ours_by_type.get(question_type),
            f"summary.by_question_type.{question_type}",
        )
        baseline_task = require_mapping(
            baseline_by_type.get(question_type),
            f"baseline.summary.by_question_type.{question_type}",
        )
        ours_accuracy = require_number(
            ours_task.get("accuracy"),
            f"summary.by_question_type.{question_type}.accuracy",
        )
        baseline_accuracy = require_number(
            baseline_task.get("accuracy"),
            f"baseline.summary.by_question_type.{question_type}.accuracy",
        )
        ours_correct = require_number(
            ours_task.get("correct"),
            f"summary.by_question_type.{question_type}.correct",
        )
        baseline_correct = require_number(
            baseline_task.get("correct"),
            f"baseline.summary.by_question_type.{question_type}.correct",
        )
        ours_total = require_number(
            ours_task.get("total"),
            f"summary.by_question_type.{question_type}.total",
        )
        baseline_total = require_number(
            baseline_task.get("total"),
            f"baseline.summary.by_question_type.{question_type}.total",
        )
        second_rows.append(
            [
                question_type,
                f"{format_percent(ours_accuracy, digits)} "
                f"({int(ours_correct)}/{int(ours_total)})",
                f"{format_percent(baseline_accuracy, digits)} "
                f"({int(baseline_correct)}/{int(baseline_total)})",
                format_delta(ours_accuracy - baseline_accuracy, digits),
            ]
        )

    headers = ["Metric", method_name(data), baseline_name(data), "Delta"]
    task_headers = ["Question task", method_name(data), baseline_name(data), "Delta"]
    return (
        markdown_table(headers, first_rows),
        markdown_table(task_headers, second_rows),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        data = load_json(args.input)
        metrics_table, tasks_table = build_tables(data, args.digits)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print("Table 1: Overall comparison")
    print(metrics_table)
    print("\nTable 2: Answer accuracy by question task")
    print(tasks_table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
