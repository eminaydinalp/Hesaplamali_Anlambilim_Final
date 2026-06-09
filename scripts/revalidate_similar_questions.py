from __future__ import annotations

import argparse
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from evaluate_openrouter_teacher import DEFAULT_MODEL
from generate_similar_questions import build_output_row, make_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-parse and compact existing similar-question rows without making "
            "new API calls."
        )
    )
    parser.add_argument("--input", default="data/train_final_500.jsonl")
    parser.add_argument("--similar-input", default="data/similar_questions.jsonl")
    parser.add_argument("--output", default="data/similar_questions.jsonl")
    parser.add_argument(
        "--summary-output",
        default="logs/similar_questions_revalidation_summary.json",
    )
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL row in {path} at line {line_number}: {error}"
                ) from error
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]], restart: bool) -> None:
    if path.exists() and not restart:
        raise FileExistsError(f"{path} already exists. Use --restart to overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any], restart: bool) -> None:
    if path.exists() and not restart:
        raise FileExistsError(f"{path} already exists. Use --restart to overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(Path(args.input))
    similar_rows = read_jsonl(Path(args.similar_input))
    source_by_id = {str(row["id"]): row for row in source_rows}
    similar_by_original_id = {
        str(row.get("original_id")): row
        for row in similar_rows
        if row.get("original_id") is not None
    }

    revalidated_rows: list[dict[str, Any]] = []
    missing_original_ids: list[str] = []
    dropped_original_ids = sorted(
        original_id
        for original_id in similar_by_original_id
        if original_id not in source_by_id
    )

    for source_row in source_rows:
        original_id = str(source_row["id"])
        existing = similar_by_original_id.get(original_id)
        if existing is None:
            missing_original_ids.append(original_id)
            continue

        model_id = str(existing.get("teacher_model_id") or DEFAULT_MODEL)
        prompt = str(existing.get("prompt") or make_prompt(source_row))
        revalidated_rows.append(
            build_output_row(
                source_row=source_row,
                model_id=model_id,
                prompt=prompt,
                model_output=existing.get("model_output"),
                generation_error=existing.get("generation_error"),
            )
        )

    valid_rows = [
        row
        for row in revalidated_rows
        if row.get("generation_error") is None
        and row.get("processing_error") is None
    ]
    error_rows = [
        row
        for row in revalidated_rows
        if row.get("generation_error") is not None
        or row.get("processing_error") is not None
    ]

    write_jsonl(Path(args.output), revalidated_rows, restart=args.restart)
    summary = {
        "input": args.input,
        "similar_input": args.similar_input,
        "output": args.output,
        "source_rows": len(source_rows),
        "previous_similar_rows": len(similar_rows),
        "revalidated_rows": len(revalidated_rows),
        "valid_rows": len(valid_rows),
        "error_rows": len(error_rows),
        "missing_rows": len(missing_original_ids),
        "missing_original_ids": missing_original_ids[:50],
        "dropped_rows": len(dropped_original_ids),
        "dropped_original_ids": dropped_original_ids[:50],
        "error_original_ids": [str(row.get("original_id")) for row in error_rows[:50]],
    }
    write_json(Path(args.summary_output), summary, restart=args.restart)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
