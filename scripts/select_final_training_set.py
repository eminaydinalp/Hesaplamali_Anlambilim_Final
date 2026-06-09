from __future__ import annotations

import argparse
import json
import random
import re
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from answer_utils import is_clock_time_target_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the final verified training set while excluding every test item."
        )
    )
    parser.add_argument("--verified-input", default="data/train_failed_verified.jsonl")
    parser.add_argument("--solutions-input", default="data/solutions.jsonl")
    parser.add_argument("--test-input", default="data/test.jsonl")
    parser.add_argument("--output", default="data/train_final_500.jsonl")
    parser.add_argument("--solutions-output", default="data/solutions_final_500.jsonl")
    parser.add_argument(
        "--summary-output",
        default="logs/final_train_selection_summary.json",
    )
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection",
        choices=["random", "first"],
        default="random",
        help="Use a reproducible random sample or the first eligible rows.",
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


def normalize_question(text: object) -> str:
    normalized = str(text or "").casefold().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def test_source_id_to_train_id(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("train-"):
        return text
    return f"train-{text}"


def deduplicate_verified_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    deduplicated: list[dict[str, Any]] = []
    duplicate_id_rows = 0
    duplicate_question_rows = 0

    for row in rows:
        row_id = str(row.get("id", "")).strip()
        question = normalize_question(row.get("question"))

        if row_id in seen_ids:
            duplicate_id_rows += 1
            continue
        if question in seen_questions:
            duplicate_question_rows += 1
            continue

        seen_ids.add(row_id)
        seen_questions.add(question)
        deduplicated.append(row)

    return deduplicated, duplicate_id_rows, duplicate_question_rows


def build_eligible_rows(
    verified_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    test_source_train_ids = {
        train_id
        for train_id in (
            test_source_id_to_train_id(row.get("source_id")) for row in test_rows
        )
        if train_id is not None
    }
    test_question_texts = {
        normalize_question(row.get("question"))
        for row in test_rows
        if normalize_question(row.get("question"))
    }

    deduplicated_rows, duplicate_id_rows, duplicate_question_rows = (
        deduplicate_verified_rows(verified_rows)
    )

    eligible_rows: list[dict[str, Any]] = []
    excluded_by_test_source_id = 0
    excluded_by_test_question = 0
    excluded_by_empty_question = 0
    excluded_by_non_scalar_answer = 0
    non_scalar_answer_ids: list[str] = []

    for row in deduplicated_rows:
        row_id = str(row.get("id", "")).strip()
        question = normalize_question(row.get("question"))

        if not question:
            excluded_by_empty_question += 1
            continue
        if row_id in test_source_train_ids:
            excluded_by_test_source_id += 1
            continue
        if question in test_question_texts:
            excluded_by_test_question += 1
            continue
        if is_clock_time_target_answer(str(row.get("answer", ""))):
            excluded_by_non_scalar_answer += 1
            non_scalar_answer_ids.append(row_id)
            continue

        eligible_rows.append(row)

    diagnostics = {
        "test_source_train_id_count": len(test_source_train_ids),
        "test_question_text_count": len(test_question_texts),
        "duplicate_verified_id_rows": duplicate_id_rows,
        "duplicate_verified_question_rows": duplicate_question_rows,
        "excluded_by_test_source_id": excluded_by_test_source_id,
        "excluded_by_test_question": excluded_by_test_question,
        "excluded_by_empty_question": excluded_by_empty_question,
        "excluded_by_non_scalar_answer": excluded_by_non_scalar_answer,
        "non_scalar_answer_ids": non_scalar_answer_ids[:50],
    }
    return eligible_rows, diagnostics


def select_rows(
    eligible_rows: list[dict[str, Any]],
    size: int,
    selection: str,
    seed: int,
) -> list[dict[str, Any]]:
    if len(eligible_rows) < size:
        raise ValueError(
            f"Only {len(eligible_rows)} eligible rows are available; {size} required."
        )

    if selection == "first":
        return eligible_rows[:size]

    sampler = random.Random(seed)
    return sampler.sample(eligible_rows, size)


def select_solution_rows(
    solution_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    solutions_by_id = {str(row.get("id")): row for row in solution_rows}
    selected_solution_rows: list[dict[str, Any]] = []
    missing_solution_ids: list[str] = []

    for row in selected_rows:
        row_id = str(row.get("id"))
        solution_row = solutions_by_id.get(row_id)
        if solution_row is None:
            missing_solution_ids.append(row_id)
        else:
            selected_solution_rows.append(solution_row)

    return selected_solution_rows, missing_solution_ids


def count_selected_overlaps(
    selected_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, int]:
    test_source_train_ids = {
        train_id
        for train_id in (
            test_source_id_to_train_id(row.get("source_id")) for row in test_rows
        )
        if train_id is not None
    }
    test_question_texts = {
        normalize_question(row.get("question"))
        for row in test_rows
        if normalize_question(row.get("question"))
    }

    selected_ids = {str(row.get("id", "")).strip() for row in selected_rows}
    selected_questions = {
        normalize_question(row.get("question"))
        for row in selected_rows
        if normalize_question(row.get("question"))
    }

    return {
        "selected_test_source_id_overlap": len(selected_ids & test_source_train_ids),
        "selected_test_question_overlap": len(selected_questions & test_question_texts),
    }


def main() -> None:
    args = parse_args()

    verified_rows = read_jsonl(Path(args.verified_input))
    solution_rows = read_jsonl(Path(args.solutions_input))
    test_rows = read_jsonl(Path(args.test_input))

    eligible_rows, diagnostics = build_eligible_rows(
        verified_rows=verified_rows,
        test_rows=test_rows,
    )
    selected_rows = select_rows(
        eligible_rows=eligible_rows,
        size=args.size,
        selection=args.selection,
        seed=args.seed,
    )
    selected_solution_rows, missing_solution_ids = select_solution_rows(
        solution_rows=solution_rows,
        selected_rows=selected_rows,
    )

    if missing_solution_ids:
        sample = ", ".join(missing_solution_ids[:10])
        raise ValueError(
            f"{len(missing_solution_ids)} selected rows are missing solutions. "
            f"Examples: {sample}"
        )

    overlap_counts = count_selected_overlaps(
        selected_rows=selected_rows,
        test_rows=test_rows,
    )
    if any(overlap_counts.values()):
        raise ValueError(f"Selected rows overlap with test set: {overlap_counts}")

    write_jsonl(Path(args.output), selected_rows, restart=args.restart)
    write_jsonl(
        Path(args.solutions_output),
        selected_solution_rows,
        restart=args.restart,
    )

    selected_ids = [str(row.get("id")) for row in selected_rows]
    summary = {
        "verified_input": args.verified_input,
        "solutions_input": args.solutions_input,
        "test_input": args.test_input,
        "output": args.output,
        "solutions_output": args.solutions_output,
        "selection": args.selection,
        "seed": args.seed,
        "requested_size": args.size,
        "test_rows": len(test_rows),
        "verified_input_rows": len(verified_rows),
        "eligible_rows": len(eligible_rows),
        "selected_rows": len(selected_rows),
        "selected_solution_rows": len(selected_solution_rows),
        "selected_ids": selected_ids,
        **diagnostics,
        **overlap_counts,
    }
    write_json(Path(args.summary_output), summary, restart=args.restart)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
