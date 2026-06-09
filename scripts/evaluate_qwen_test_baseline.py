from __future__ import annotations

import argparse
import json
import os
from json import JSONDecodeError
from pathlib import Path
from time import time

from tqdm.auto import tqdm

from answer_utils import canonical_number, extract_model_final_answer, numbers_match
from evaluate_m1_failures import (
    generate_batch_with_fallback,
    load_model,
    make_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the base Qwen model on the fixed test set."
    )
    parser.add_argument("--model-id", default=os.getenv("M1_MODEL_ID", "Qwen/Qwen3.5-4B"))
    parser.add_argument("--test-input", default="data/test.jsonl")
    parser.add_argument(
        "--references-input",
        default="data/test_with_reference_numeric.jsonl",
    )
    parser.add_argument(
        "--predictions-output",
        default="data/test_qwen_predictions.jsonl",
    )
    parser.add_argument(
        "--failed-output",
        default="data/test_qwen_failed.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        default="logs/qwen_test_baseline_summary.json",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_predictions(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}

    predictions: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except JSONDecodeError:
                continue
            predictions[str(row["id"])] = row
    return predictions


def merge_test_rows(
    test_rows: list[dict[str, object]],
    reference_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    references_by_id = {
        str(row["question_id"]): row for row in reference_rows
    }
    rows: list[dict[str, object]] = []
    for row in test_rows:
        question_id = str(row["question_id"])
        reference_row = references_by_id.get(question_id, {})
        reference_answer_raw = reference_row.get("reference_numeric_answer")
        rows.append(
            {
                "id": question_id,
                "source_id": row.get("source_id"),
                "source_split": row.get("split"),
                "question": row["question"],
                "answer": row.get("reference_answer"),
                "reference_answer_raw": (
                    str(reference_answer_raw)
                    if reference_answer_raw is not None
                    else None
                ),
                "reference_answer": canonical_number(
                    str(reference_answer_raw)
                    if reference_answer_raw is not None
                    else None
                ),
            }
        )
    return rows


def parse_prediction(
    model_output: str | None,
    reference_answer_raw: str | None,
) -> tuple[str | None, str | None, bool, str | None]:
    try:
        predicted_answer_raw = extract_model_final_answer(model_output)
        predicted_answer = canonical_number(predicted_answer_raw)
        is_correct = numbers_match(predicted_answer_raw, reference_answer_raw)
        return predicted_answer_raw, predicted_answer, is_correct, None
    except Exception as error:
        return None, None, False, f"{type(error).__name__}: {error}"


def write_outputs(
    rows: list[dict[str, object]],
    predictions_by_id: dict[str, dict[str, object]],
    predictions_output: Path,
    failed_output: Path,
    summary_output: Path,
    model_id: str,
    device: str,
    started_at: float,
) -> None:
    ordered_predictions = [
        predictions_by_id[str(row["id"])]
        for row in rows
        if str(row["id"]) in predictions_by_id
    ]
    scored_rows = [
        row
        for row in ordered_predictions
        if row.get("generation_error") is None
        and row.get("processing_error") is None
    ]
    failed_rows = [
        row for row in scored_rows if row.get("is_correct") is False
    ]
    correct_rows = [
        row for row in scored_rows if row.get("is_correct") is True
    ]
    error_rows = [
        row
        for row in ordered_predictions
        if row.get("generation_error") is not None
        or row.get("processing_error") is not None
    ]
    model_parse_failed_rows = [
        row
        for row in scored_rows
        if row.get("predicted_answer_raw") is None
    ]

    write_jsonl(failed_output, failed_rows)
    summary = {
        "model_id": model_id,
        "device": device,
        "total_input_rows": len(rows),
        "evaluated_rows": len(ordered_predictions),
        "scored_rows": len(scored_rows),
        "correct_rows": len(correct_rows),
        "failed_rows": len(failed_rows),
        "error_rows": len(error_rows),
        "model_parse_failed_rows": len(model_parse_failed_rows),
        "accuracy": len(correct_rows) / len(scored_rows) if scored_rows else None,
        "predictions_output": str(predictions_output),
        "failed_output": str(failed_output),
        "elapsed_seconds": round(time() - started_at, 2),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    started_at = time()

    test_rows = read_jsonl(Path(args.test_input))
    reference_rows = read_jsonl(Path(args.references_input))
    rows = merge_test_rows(test_rows, reference_rows)
    if args.limit is not None:
        rows = rows[: args.limit]

    predictions_output = Path(args.predictions_output)
    failed_output = Path(args.failed_output)
    summary_output = Path(args.summary_output)
    predictions_by_id = (
        {} if args.restart else load_existing_predictions(predictions_output)
    )
    pending_rows = [
        row for row in rows if str(row["id"]) not in predictions_by_id
    ]

    _, tokenizer, model, device = load_model(args.model_id, args.allow_download)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if args.restart else "a"
    with predictions_output.open(mode, encoding="utf-8") as output_file:
        progress = tqdm(
            range(0, len(pending_rows), args.batch_size),
            desc="Qwen test baseline",
        )
        for batch_index, start in enumerate(progress, start=1):
            batch_rows = pending_rows[start : start + args.batch_size]
            prompts = [make_prompt(str(row["question"])) for row in batch_rows]
            outputs = generate_batch_with_fallback(
                tokenizer=tokenizer,
                model=model,
                prompts=prompts,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
            )

            for row, prompt, output_result in zip(batch_rows, prompts, outputs):
                model_output, generation_error = output_result
                predicted_answer_raw, predicted_answer, is_correct, processing_error = (
                    parse_prediction(model_output, row["reference_answer_raw"])
                    if generation_error is None
                    else (None, None, False, None)
                )
                prediction = {
                    "id": row["id"],
                    "source_id": row.get("source_id"),
                    "source_split": row.get("source_split"),
                    "model_id": args.model_id,
                    "model_type": "qwen3_5",
                    "question": row["question"],
                    "answer": row.get("answer"),
                    "reference_answer_raw": row.get("reference_answer_raw"),
                    "reference_answer": row.get("reference_answer"),
                    "prompt": prompt,
                    "model_output": model_output.strip() if model_output else None,
                    "predicted_answer_raw": predicted_answer_raw,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "generation_error": generation_error,
                    "processing_error": processing_error,
                }
                predictions_by_id[str(row["id"])] = prediction
                output_file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                output_file.flush()

            if batch_index % args.checkpoint_every == 0:
                write_outputs(
                    rows=rows,
                    predictions_by_id=predictions_by_id,
                    predictions_output=predictions_output,
                    failed_output=failed_output,
                    summary_output=summary_output,
                    model_id=args.model_id,
                    device=device,
                    started_at=started_at,
                )

    write_outputs(
        rows=rows,
        predictions_by_id=predictions_by_id,
        predictions_output=predictions_output,
        failed_output=failed_output,
        summary_output=summary_output,
        model_id=args.model_id,
        device=device,
        started_at=started_at,
    )
    print(summary_output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
