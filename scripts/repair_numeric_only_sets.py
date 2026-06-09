from __future__ import annotations

import argparse
import json
import random
import re
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from answer_utils import (
    canonical_number,
    extract_last_answer_token,
    extract_scalar_reference_answer,
    is_clock_time_target_answer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair the fixed test set and final train set so both contain only "
            "scalar numeric target answers."
        )
    )
    parser.add_argument("--source-input", default="data/gsm8k_tr.jsonl")
    parser.add_argument("--test-input", default="data/test.jsonl")
    parser.add_argument(
        "--test-references-output",
        default="data/test_with_reference_numeric.jsonl",
    )
    parser.add_argument("--verified-input", default="data/train_failed_verified.jsonl")
    parser.add_argument("--solutions-input", default="data/solutions.jsonl")
    parser.add_argument("--train-input", default="data/train_final_500.jsonl")
    parser.add_argument("--train-output", default="data/train_final_500.jsonl")
    parser.add_argument("--solutions-output", default="data/solutions_final_500.jsonl")
    parser.add_argument(
        "--summary-output",
        default="logs/numeric_only_repair_summary.json",
    )
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
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
    return re.sub(r"\s+", " ", normalized)


def source_id_to_train_id(value: object) -> str:
    text = str(value).strip()
    return text if text.startswith("train-") else f"train-{text}"


def train_id_to_source_id(value: object) -> int:
    text = str(value).strip()
    if not text.startswith("train-"):
        raise ValueError(f"Expected a train-* id, got: {value}")
    return int(text.split("-", maxsplit=1)[1])


def scalar_reference_from_solution(answer_text: object) -> tuple[str | None, str | None]:
    raw = extract_scalar_reference_answer(str(answer_text or ""))
    return raw, canonical_number(raw)


def answer_exclusion_reason(answer_text: object) -> str | None:
    text = str(answer_text or "")
    if is_clock_time_target_answer(text):
        return "clock_time_target_answer"

    _, reference = scalar_reference_from_solution(text)
    if reference is None:
        return "reference_parse_failed"
    return None


def describe_excluded_row(
    row: dict[str, Any],
    reason: str,
    id_key: str,
) -> dict[str, Any]:
    token = extract_last_answer_token(
        row.get("answer") or row.get("reference_answer") or ""
    )
    return {
        "id": row.get(id_key),
        "source_id": row.get("source_id"),
        "reason": reason,
        "last_answer_token_type": token[0] if token else None,
        "last_answer_token": token[1] if token else None,
    }


def build_test_reference_rows(test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference_rows: list[dict[str, Any]] = []
    for row in test_rows:
        raw, reference = scalar_reference_from_solution(row.get("reference_answer"))
        if reference is None:
            raise ValueError(
                f"Cannot extract scalar reference for test row {row.get('question_id')}"
            )
        reference_rows.append(
            {
                "question_id": row["question_id"],
                "source_id": row["source_id"],
                "split": row.get("split", "test"),
                "reference_numeric_answer": reference,
            }
        )
    return reference_rows


def make_test_row(source_row: dict[str, Any], question_id: str) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "source_id": train_id_to_source_id(source_row["id"]),
        "split": "test",
        "question": source_row["question"],
        "reference_answer": source_row["answer"],
    }


def repair_test_rows(
    source_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    reserved_train_rows: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    invalid: list[tuple[int, str]] = []
    for index, row in enumerate(test_rows):
        reason = answer_exclusion_reason(row.get("reference_answer"))
        if reason is not None:
            invalid.append((index, reason))

    if not invalid:
        return test_rows, [], [], 0

    current_source_ids = {str(row.get("source_id")) for row in test_rows}
    current_questions = {
        normalize_question(row.get("question"))
        for row in test_rows
        if normalize_question(row.get("question"))
    }
    reserved_train_ids = {str(row.get("id")) for row in reserved_train_rows}
    reserved_train_questions = {
        normalize_question(row.get("question"))
        for row in reserved_train_rows
        if normalize_question(row.get("question"))
    }

    candidates: list[dict[str, Any]] = []
    for row in source_rows:
        row_id = str(row.get("id"))
        source_id = str(train_id_to_source_id(row_id))
        question = normalize_question(row.get("question"))
        if source_id in current_source_ids:
            continue
        if row_id in reserved_train_ids:
            continue
        if question in current_questions or question in reserved_train_questions:
            continue
        if answer_exclusion_reason(row.get("answer")) is not None:
            continue
        candidates.append(row)

    if len(candidates) < len(invalid):
        raise ValueError(
            f"Only {len(candidates)} clean test replacement candidates are available; "
            f"{len(invalid)} required."
        )

    replacements = random.Random(seed).sample(candidates, len(invalid))
    repaired_rows = list(test_rows)
    removed_rows: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []

    for (index, reason), replacement in zip(invalid, replacements):
        old_row = repaired_rows[index]
        new_row = make_test_row(replacement, str(old_row["question_id"]))
        removed_rows.append(describe_excluded_row(old_row, reason, "question_id"))
        replacement_rows.append(
            {
                "question_id": new_row["question_id"],
                "source_id": new_row["source_id"],
                "source_train_id": replacement["id"],
            }
        )
        repaired_rows[index] = new_row

    return repaired_rows, removed_rows, replacement_rows, len(candidates)


def row_has_test_overlap(
    row: dict[str, Any],
    test_train_ids: set[str],
    test_questions: set[str],
) -> bool:
    return (
        str(row.get("id")) in test_train_ids
        or normalize_question(row.get("question")) in test_questions
    )


def repair_train_rows(
    verified_rows: list[dict[str, Any]],
    solution_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    solutions_by_id = {str(row.get("id")): row for row in solution_rows}
    test_train_ids = {source_id_to_train_id(row.get("source_id")) for row in test_rows}
    test_questions = {
        normalize_question(row.get("question"))
        for row in test_rows
        if normalize_question(row.get("question"))
    }

    repaired_rows: list[dict[str, Any] | None] = list(train_rows)
    invalid: list[tuple[int, str]] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()

    for index, row in enumerate(train_rows):
        row_id = str(row.get("id"))
        question = normalize_question(row.get("question"))
        reason = answer_exclusion_reason(row.get("answer"))
        if reason is None and row_has_test_overlap(row, test_train_ids, test_questions):
            reason = "test_overlap"
        if reason is None and row_id not in solutions_by_id:
            reason = "missing_solution"
        if reason is None and row_id in seen_ids:
            reason = "duplicate_train_id"
        if reason is None and question in seen_questions:
            reason = "duplicate_train_question"

        if reason is not None:
            invalid.append((index, reason))
            repaired_rows[index] = None
            continue

        seen_ids.add(row_id)
        seen_questions.add(question)

    if not invalid:
        return train_rows, [], [], 0

    candidates: list[dict[str, Any]] = []
    candidate_seen_ids: set[str] = set()
    candidate_seen_questions: set[str] = set()
    for row in verified_rows:
        row_id = str(row.get("id"))
        question = normalize_question(row.get("question"))
        if row_id in seen_ids or row_id in candidate_seen_ids:
            continue
        if question in seen_questions or question in candidate_seen_questions:
            continue
        if row_has_test_overlap(row, test_train_ids, test_questions):
            continue
        if answer_exclusion_reason(row.get("answer")) is not None:
            continue
        if row_id not in solutions_by_id:
            continue
        candidates.append(row)
        candidate_seen_ids.add(row_id)
        candidate_seen_questions.add(question)

    if len(candidates) < len(invalid):
        raise ValueError(
            f"Only {len(candidates)} clean train replacement candidates are available; "
            f"{len(invalid)} required."
        )

    replacements = random.Random(seed).sample(candidates, len(invalid))
    removed_rows: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []

    for (index, reason), replacement in zip(invalid, replacements):
        old_row = train_rows[index]
        removed_rows.append(describe_excluded_row(old_row, reason, "id"))
        replacement_rows.append(
            {
                "id": replacement["id"],
                "replaced_index": index,
                "replaced_id": old_row.get("id"),
            }
        )
        repaired_rows[index] = replacement

    return [row for row in repaired_rows if row is not None], removed_rows, replacement_rows, len(candidates)


def select_solution_rows(
    solution_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    solutions_by_id = {str(row.get("id")): row for row in solution_rows}
    selected: list[dict[str, Any]] = []
    missing: list[str] = []

    for row in train_rows:
        row_id = str(row.get("id"))
        solution = solutions_by_id.get(row_id)
        if solution is None:
            missing.append(row_id)
        else:
            selected.append(solution)

    if missing:
        sample = ", ".join(missing[:10])
        raise ValueError(f"{len(missing)} selected train rows miss solutions: {sample}")
    return selected


def count_train_test_overlap(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, int]:
    train_ids = {str(row.get("id")) for row in train_rows}
    train_questions = {
        normalize_question(row.get("question"))
        for row in train_rows
        if normalize_question(row.get("question"))
    }
    test_train_ids = {source_id_to_train_id(row.get("source_id")) for row in test_rows}
    test_questions = {
        normalize_question(row.get("question"))
        for row in test_rows
        if normalize_question(row.get("question"))
    }
    return {
        "train_test_source_id_overlap": len(train_ids & test_train_ids),
        "train_test_question_overlap": len(train_questions & test_questions),
    }


def count_non_scalar_rows(rows: list[dict[str, Any]], answer_key: str) -> int:
    return sum(
        1
        for row in rows
        if answer_exclusion_reason(row.get(answer_key)) is not None
    )


def main() -> None:
    args = parse_args()

    source_rows = read_jsonl(Path(args.source_input))
    test_rows = read_jsonl(Path(args.test_input))
    verified_rows = read_jsonl(Path(args.verified_input))
    solution_rows = read_jsonl(Path(args.solutions_input))
    train_rows = read_jsonl(Path(args.train_input))

    repaired_test_rows, removed_test_rows, test_replacements, test_candidate_count = (
        repair_test_rows(
            source_rows=source_rows,
            test_rows=test_rows,
            reserved_train_rows=train_rows,
            seed=args.seed,
        )
    )
    repaired_test_reference_rows = build_test_reference_rows(repaired_test_rows)
    repaired_train_rows, removed_train_rows, train_replacements, train_candidate_count = (
        repair_train_rows(
            verified_rows=verified_rows,
            solution_rows=solution_rows,
            train_rows=train_rows,
            test_rows=repaired_test_rows,
            seed=args.seed,
        )
    )
    repaired_solution_rows = select_solution_rows(
        solution_rows=solution_rows,
        train_rows=repaired_train_rows,
    )

    if len(repaired_test_rows) != args.test_size:
        raise ValueError(
            f"Repaired test set has {len(repaired_test_rows)} rows; "
            f"{args.test_size} expected."
        )
    if len(repaired_train_rows) != args.train_size:
        raise ValueError(
            f"Repaired train set has {len(repaired_train_rows)} rows; "
            f"{args.train_size} expected."
        )

    overlap_counts = count_train_test_overlap(repaired_train_rows, repaired_test_rows)
    if any(overlap_counts.values()):
        raise ValueError(f"Repaired train/test sets still overlap: {overlap_counts}")

    test_non_scalar_rows = count_non_scalar_rows(
        repaired_test_rows,
        answer_key="reference_answer",
    )
    train_non_scalar_rows = count_non_scalar_rows(
        repaired_train_rows,
        answer_key="answer",
    )
    if test_non_scalar_rows or train_non_scalar_rows:
        raise ValueError(
            "Repaired sets still contain non-scalar answers: "
            f"test={test_non_scalar_rows}, train={train_non_scalar_rows}"
        )

    write_jsonl(Path(args.test_input), repaired_test_rows, restart=args.restart)
    write_jsonl(
        Path(args.test_references_output),
        repaired_test_reference_rows,
        restart=args.restart,
    )
    write_jsonl(Path(args.train_output), repaired_train_rows, restart=args.restart)
    write_jsonl(Path(args.solutions_output), repaired_solution_rows, restart=args.restart)

    summary = {
        "source_input": args.source_input,
        "test_input": args.test_input,
        "test_references_output": args.test_references_output,
        "verified_input": args.verified_input,
        "solutions_input": args.solutions_input,
        "train_input": args.train_input,
        "train_output": args.train_output,
        "solutions_output": args.solutions_output,
        "seed": args.seed,
        "test_rows": len(repaired_test_rows),
        "train_rows": len(repaired_train_rows),
        "solution_rows": len(repaired_solution_rows),
        "removed_test_rows": removed_test_rows,
        "test_replacements": test_replacements,
        "test_replacement_candidate_count": test_candidate_count,
        "removed_train_rows": removed_train_rows,
        "train_replacements": train_replacements,
        "train_replacement_candidate_count": train_candidate_count,
        "test_non_scalar_rows_after_repair": test_non_scalar_rows,
        "train_non_scalar_rows_after_repair": train_non_scalar_rows,
        **overlap_counts,
    }
    write_json(Path(args.summary_output), summary, restart=args.restart)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
