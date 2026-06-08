from __future__ import annotations

import argparse
import json
import os
from json import JSONDecodeError
from pathlib import Path
from time import time

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

from answer_utils import (
    canonical_number,
    extract_model_final_answer,
    numbers_match,
)


PROMPT_TEMPLATE = """Soru:
{question}

Bu problemi kısaca açıklayarak Türkçe çöz. Gereksiz açıklama yapma, 2-5 kısa hesap adımı yaz.
Cevabının son satırında yalnızca şu formatı kullan:
Final cevap: <sayı>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate M1 on GSM8K_TR and collect failed examples."
    )
    parser.add_argument("--model-id", default=os.getenv("M1_MODEL_ID", "Qwen/Qwen3.5-4B"))
    parser.add_argument("--questions-input", default="data/gsm8k_tr.jsonl")
    parser.add_argument("--references-input", default="data/gsm8k_tr_references.jsonl")
    parser.add_argument("--predictions-output", default="data/m1_predictions.jsonl")
    parser.add_argument("--failed-output", default="data/train_failed.jsonl")
    parser.add_argument("--summary-output", default="logs/m1_eval_summary.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


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


def make_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


def load_references(path: Path) -> dict[str, dict[str, object]]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def merge_questions_with_references(
    question_rows: list[dict[str, object]],
    references_by_id: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in question_rows:
        row_id = str(row["id"])
        reference = references_by_id.get(row_id, {})
        rows.append(
            {
                **row,
                "reference_answer_raw": reference.get("reference_answer_raw"),
                "reference_answer": reference.get("reference_answer"),
            }
        )
    return rows


def prepare_reference(row: dict[str, object]) -> tuple[str | None, str | None]:
    raw = row.get("reference_answer_raw")
    normalized = row.get("reference_answer")
    if raw is None and normalized is None:
        return None, None
    if normalized is None:
        normalized = canonical_number(str(raw))
    return str(raw) if raw is not None else None, str(normalized) if normalized is not None else None


def load_model(model_id: str, allow_download: bool):
    local_files_only = not allow_download
    device = choose_device()
    config = AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto" if device == "cuda" else None,
        local_files_only=local_files_only,
    )
    model.eval()
    return config, tokenizer, model, device


def generate_batch(
    tokenizer,
    model,
    prompts: list[str],
    max_input_tokens: int,
    max_new_tokens: int,
) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[:, input_length:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def generate_batch_with_fallback(
    tokenizer,
    model,
    prompts: list[str],
    max_input_tokens: int,
    max_new_tokens: int,
) -> list[tuple[str | None, str | None]]:
    try:
        outputs = generate_batch(
            tokenizer=tokenizer,
            model=model,
            prompts=prompts,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )
        return [(output, None) for output in outputs]
    except RuntimeError as error:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results: list[tuple[str | None, str | None]] = []
        for prompt in prompts:
            try:
                output = generate_batch(
                    tokenizer=tokenizer,
                    model=model,
                    prompts=[prompt],
                    max_input_tokens=max_input_tokens,
                    max_new_tokens=max_new_tokens,
                )[0]
                results.append((output, None))
            except Exception as single_error:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                results.append((None, f"{type(single_error).__name__}: {single_error}"))
        return results
    except Exception as error:
        return [(None, f"{type(error).__name__}: {error}") for _ in prompts]


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


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(
    rows: list[dict[str, object]],
    predictions_by_id: dict[str, dict[str, object]],
    predictions_output: Path,
    failed_output: Path,
    summary_output: Path,
    model_id: str,
    device: str,
    started_at: float,
    skipped_reference_parse_failed_rows: int,
) -> None:
    ordered_predictions = [
        predictions_by_id[str(row["id"])]
        for row in rows
        if str(row["id"]) in predictions_by_id
    ]
    failed_rows = [
        prediction
        for prediction in ordered_predictions
        if prediction["is_correct"] is False
        and prediction.get("generation_error") is None
        and prediction.get("processing_error") is None
    ]
    model_parse_failed_rows = [
        prediction
        for prediction in ordered_predictions
        if prediction["predicted_answer_raw"] is None
        and prediction.get("generation_error") is None
        and prediction.get("processing_error") is None
    ]
    error_rows = [
        prediction
        for prediction in ordered_predictions
        if prediction.get("generation_error") is not None
        or prediction.get("processing_error") is not None
    ]
    scored_rows = [
        prediction
        for prediction in ordered_predictions
        if prediction.get("generation_error") is None
        and prediction.get("processing_error") is None
    ]
    correct_rows = [
        prediction for prediction in scored_rows if prediction["is_correct"] is True
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
        "skipped_reference_parse_failed_rows": skipped_reference_parse_failed_rows,
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
    questions_input = Path(args.questions_input)
    references_input = Path(args.references_input)
    predictions_output = Path(args.predictions_output)
    failed_output = Path(args.failed_output)
    summary_output = Path(args.summary_output)

    question_rows = read_jsonl(questions_input)
    references_by_id = load_references(references_input)
    all_rows = merge_questions_with_references(question_rows, references_by_id)
    if args.limit is not None:
        all_rows = all_rows[: args.limit]

    skipped_rows = [
        row for row in all_rows if row.get("reference_answer_raw") is None
    ]
    rows = [
        row for row in all_rows if row.get("reference_answer_raw") is not None
    ]

    predictions_by_id = (
        {} if args.restart else load_existing_predictions(predictions_output)
    )
    pending_rows = [
        row for row in rows if str(row["id"]) not in predictions_by_id
    ]

    config, tokenizer, model, device = load_model(args.model_id, args.allow_download)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if args.restart else "a"
    with predictions_output.open(mode, encoding="utf-8") as output_file:
        progress = tqdm(
            range(0, len(pending_rows), args.batch_size),
            desc="M1 evaluating",
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
                reference_answer_raw, reference_answer = prepare_reference(row)
                predicted_answer_raw, predicted_answer, is_correct, processing_error = (
                    parse_prediction(model_output, reference_answer_raw)
                    if generation_error is None
                    else (None, None, False, None)
                )

                prediction = {
                    "id": row["id"],
                    "source_dataset": row.get("source_dataset"),
                    "source_split": row.get("source_split"),
                    "model_id": args.model_id,
                    "model_type": getattr(config, "model_type", None),
                    "question": row["question"],
                    "answer": row.get("answer"),
                    "reference_answer_raw": reference_answer_raw,
                    "reference_answer": reference_answer,
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
                    skipped_reference_parse_failed_rows=len(skipped_rows),
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
        skipped_reference_parse_failed_rows=len(skipped_rows),
    )

    print(summary_output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
