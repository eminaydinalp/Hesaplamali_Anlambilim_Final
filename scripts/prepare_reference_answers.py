from __future__ import annotations

import argparse
import json
from pathlib import Path

from answer_utils import canonical_number, extract_scalar_reference_answer, is_clock_time_target_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract reference final answers from GSM8K_TR answers."
    )
    parser.add_argument("--input", default="data/gsm8k_tr.jsonl")
    parser.add_argument("--output", default="data/gsm8k_tr_references.jsonl")
    parser.add_argument("--summary-output", default="logs/reference_extraction_summary.json")
    parser.add_argument("--failures-output", default="logs/reference_extraction_failures.jsonl")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    failures_path = Path(args.failures_output)

    rows = read_jsonl(input_path)
    output_rows: list[dict[str, object]] = []
    parse_failures: list[dict[str, object]] = []
    non_scalar_rows: list[dict[str, object]] = []

    for row in rows:
        answer_text = str(row.get("answer", ""))
        is_non_scalar = is_clock_time_target_answer(answer_text)
        reference_answer_raw = extract_scalar_reference_answer(answer_text)
        reference_answer = canonical_number(reference_answer_raw)
        answer_type = "clock_time" if is_non_scalar else "scalar_numeric"
        if is_non_scalar:
            non_scalar_rows.append(
                {
                    "id": row.get("id"),
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                }
            )
        elif reference_answer is None:
            parse_failures.append(
                {
                    "id": row.get("id"),
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                }
            )

        output_rows.append(
            {
                "id": row.get("id"),
                "reference_answer_raw": reference_answer_raw,
                "reference_answer": reference_answer,
                "answer_type": answer_type,
            }
        )

    write_jsonl(output_path, output_rows)
    write_jsonl(failures_path, parse_failures)
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "failures_path": str(failures_path),
        "total_rows": len(rows),
        "parsed_rows": len(rows) - len(parse_failures) - len(non_scalar_rows),
        "parse_failure_count": len(parse_failures),
        "non_scalar_answer_count": len(non_scalar_rows),
        "parse_failure_ids": [str(row["id"]) for row in parse_failures[:50]],
        "non_scalar_answer_ids": [str(row["id"]) for row in non_scalar_rows[:50]],
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
