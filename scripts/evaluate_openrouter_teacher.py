from __future__ import annotations

import argparse
import json
import os
from json import JSONDecodeError
from pathlib import Path
from time import sleep, time

from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm

from answer_utils import canonical_number, extract_model_final_answer, numbers_match
from evaluate_m1_failures import make_prompt


DEFAULT_MODEL = "openai/gpt-oss-120b:free"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask an OpenRouter teacher model to verify M1 failed examples."
    )
    parser.add_argument("--input", default="data/train_failed.jsonl")
    parser.add_argument(
        "--predictions-output",
        default="data/gpt_oss_120b_predictions.jsonl",
    )
    parser.add_argument(
        "--verified-output",
        default="data/train_failed_verified.jsonl",
    )
    parser.add_argument(
        "--disputed-output",
        default="data/train_failed_disputed.jsonl",
    )
    parser.add_argument("--solutions-output", default="data/solutions.jsonl")
    parser.add_argument(
        "--summary-output",
        default="logs/openrouter_teacher_summary.json",
    )
    parser.add_argument("--model-id", default=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep-seconds", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


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


def build_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Add it to .env before running this script."
        )

    default_headers: dict[str, str] = {}
    site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
    app_name = os.getenv("OPENROUTER_APP_NAME", "").strip()
    if site_url:
        default_headers["HTTP-Referer"] = site_url
    if app_name:
        default_headers["X-Title"] = app_name

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers=default_headers or None,
    )


def call_teacher(
    client: OpenAI,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int | None,
    max_retries: int,
    retry_sleep_seconds: float,
) -> tuple[str | None, str | None]:
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            request_payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if seed is not None:
                request_payload["seed"] = seed
            response = client.chat.completions.create(**request_payload)
            if not response.choices:
                if hasattr(response, "model_dump"):
                    response_payload = response.model_dump(mode="json")
                else:
                    response_payload = str(response)
                raise RuntimeError(
                    "OpenRouter returned no choices: "
                    + json.dumps(response_payload, ensure_ascii=False)[:1000]
                )
            content = response.choices[0].message.content
            if isinstance(content, list):
                text = "\n".join(
                    str(part.get("text", part)) if isinstance(part, dict) else str(part)
                    for part in content
                )
            else:
                text = content or ""
            return text.strip(), None
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < max_retries:
                sleep(retry_sleep_seconds * (attempt + 1))
    return None, last_error


def parse_teacher_prediction(
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


def build_teacher_prediction(
    row: dict[str, object],
    model_id: str,
    prompt: str,
    model_output: str | None,
    generation_error: str | None,
) -> dict[str, object]:
    reference_answer_raw = (
        str(row["reference_answer_raw"])
        if row.get("reference_answer_raw") is not None
        else None
    )
    reference_answer = (
        str(row["reference_answer"]) if row.get("reference_answer") is not None else None
    )
    predicted_answer_raw, predicted_answer, is_correct, processing_error = (
        parse_teacher_prediction(model_output, reference_answer_raw)
        if generation_error is None
        else (None, None, False, None)
    )

    return {
        "id": row["id"],
        "source_dataset": row.get("source_dataset"),
        "source_split": row.get("source_split"),
        "model_id": model_id,
        "model_type": "openrouter",
        "question": row["question"],
        "answer": row.get("answer"),
        "reference_answer_raw": reference_answer_raw,
        "reference_answer": reference_answer,
        "prompt": prompt,
        "model_output": model_output,
        "predicted_answer_raw": predicted_answer_raw,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "generation_error": generation_error,
        "processing_error": processing_error,
    }


def qwen_disagrees_with_teacher(
    qwen_row: dict[str, object],
    teacher_row: dict[str, object],
) -> bool:
    teacher_answer = teacher_row.get("predicted_answer_raw")
    qwen_answer = qwen_row.get("predicted_answer_raw")
    if teacher_answer is None:
        return False
    if qwen_answer is None:
        return True
    return not numbers_match(str(qwen_answer), str(teacher_answer))


def build_verified_row(
    qwen_row: dict[str, object],
    teacher_row: dict[str, object],
) -> dict[str, object]:
    return {
        "id": qwen_row["id"],
        "source_dataset": qwen_row.get("source_dataset"),
        "source_split": qwen_row.get("source_split"),
        "question": qwen_row["question"],
        "answer": qwen_row.get("answer"),
        "reference_answer_raw": qwen_row.get("reference_answer_raw"),
        "reference_answer": qwen_row.get("reference_answer"),
        "m1_model_id": qwen_row.get("model_id"),
        "m1_model_output": qwen_row.get("model_output"),
        "m1_predicted_answer_raw": qwen_row.get("predicted_answer_raw"),
        "m1_predicted_answer": qwen_row.get("predicted_answer"),
        "teacher_model_id": teacher_row.get("model_id"),
        "teacher_model_output": teacher_row.get("model_output"),
        "teacher_predicted_answer_raw": teacher_row.get("predicted_answer_raw"),
        "teacher_predicted_answer": teacher_row.get("predicted_answer"),
        "teacher_is_correct": teacher_row.get("is_correct"),
        "m1_teacher_answers_match": not qwen_disagrees_with_teacher(
            qwen_row,
            teacher_row,
        ),
    }


def build_solution_row(verified_row: dict[str, object]) -> dict[str, object]:
    return {
        "id": verified_row["id"],
        "question": verified_row["question"],
        "reference_answer": verified_row.get("reference_answer"),
        "teacher_model_id": verified_row.get("teacher_model_id"),
        "solution": verified_row.get("teacher_model_output"),
        "teacher_predicted_answer": verified_row.get("teacher_predicted_answer"),
    }


def write_outputs(
    rows: list[dict[str, object]],
    teacher_predictions_by_id: dict[str, dict[str, object]],
    predictions_output: Path,
    verified_output: Path,
    disputed_output: Path,
    solutions_output: Path,
    summary_output: Path,
    model_id: str,
    started_at: float,
) -> None:
    ordered_teacher_predictions = [
        teacher_predictions_by_id[str(row["id"])]
        for row in rows
        if str(row["id"]) in teacher_predictions_by_id
    ]
    qwen_by_id = {str(row["id"]): row for row in rows}

    verified_rows: list[dict[str, object]] = []
    disputed_rows: list[dict[str, object]] = []
    error_rows = 0
    teacher_correct_rows = 0
    teacher_wrong_rows = 0
    teacher_parse_failed_rows = 0

    for teacher_row in ordered_teacher_predictions:
        qwen_row = qwen_by_id[str(teacher_row["id"])]
        has_error = (
            teacher_row.get("generation_error") is not None
            or teacher_row.get("processing_error") is not None
        )
        if has_error:
            error_rows += 1
            disputed_rows.append(build_verified_row(qwen_row, teacher_row))
            continue

        if teacher_row.get("predicted_answer_raw") is None:
            teacher_parse_failed_rows += 1

        if teacher_row.get("is_correct") is True:
            teacher_correct_rows += 1
            if qwen_disagrees_with_teacher(qwen_row, teacher_row):
                verified_rows.append(build_verified_row(qwen_row, teacher_row))
            else:
                disputed_rows.append(build_verified_row(qwen_row, teacher_row))
        else:
            teacher_wrong_rows += 1
            disputed_rows.append(build_verified_row(qwen_row, teacher_row))

    write_jsonl(verified_output, verified_rows)
    write_jsonl(disputed_output, disputed_rows)
    write_jsonl(solutions_output, [build_solution_row(row) for row in verified_rows])

    summary = {
        "teacher_model_id": model_id,
        "input_rows": len(rows),
        "evaluated_rows": len(ordered_teacher_predictions),
        "teacher_reference_correct_rows": teacher_correct_rows,
        "teacher_reference_wrong_rows": teacher_wrong_rows,
        "verified_failed_rows": len(verified_rows),
        "disputed_rows": len(disputed_rows),
        "error_rows": error_rows,
        "teacher_parse_failed_rows": teacher_parse_failed_rows,
        "predictions_output": str(predictions_output),
        "verified_output": str(verified_output),
        "disputed_output": str(disputed_output),
        "solutions_output": str(solutions_output),
        "elapsed_seconds": round(time() - started_at, 2),
    }

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    started_at = time()

    input_path = Path(args.input)
    predictions_output = Path(args.predictions_output)
    verified_output = Path(args.verified_output)
    disputed_output = Path(args.disputed_output)
    solutions_output = Path(args.solutions_output)
    summary_output = Path(args.summary_output)

    rows = read_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    teacher_predictions_by_id = (
        {} if args.restart else load_existing_predictions(predictions_output)
    )
    pending_rows = [
        row for row in rows if str(row["id"]) not in teacher_predictions_by_id
    ]

    client = build_client()
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.restart else "a"

    with predictions_output.open(mode, encoding="utf-8") as output_file:
        progress = tqdm(pending_rows, desc="Teacher verifying")
        for index, row in enumerate(progress, start=1):
            prompt = make_prompt(str(row["question"]))
            model_output, generation_error = call_teacher(
                client=client,
                model_id=args.model_id,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.seed,
                max_retries=args.max_retries,
                retry_sleep_seconds=args.retry_sleep_seconds,
            )
            teacher_prediction = build_teacher_prediction(
                row=row,
                model_id=args.model_id,
                prompt=prompt,
                model_output=model_output,
                generation_error=generation_error,
            )
            teacher_predictions_by_id[str(row["id"])] = teacher_prediction
            output_file.write(json.dumps(teacher_prediction, ensure_ascii=False) + "\n")
            output_file.flush()

            if index % args.checkpoint_every == 0:
                write_outputs(
                    rows=rows,
                    teacher_predictions_by_id=teacher_predictions_by_id,
                    predictions_output=predictions_output,
                    verified_output=verified_output,
                    disputed_output=disputed_output,
                    solutions_output=solutions_output,
                    summary_output=summary_output,
                    model_id=args.model_id,
                    started_at=started_at,
                )

            if args.sleep_seconds > 0:
                sleep(args.sleep_seconds)

    write_outputs(
        rows=rows,
        teacher_predictions_by_id=teacher_predictions_by_id,
        predictions_output=predictions_output,
        verified_output=verified_output,
        disputed_output=disputed_output,
        solutions_output=solutions_output,
        summary_output=summary_output,
        model_id=args.model_id,
        started_at=started_at,
    )
    print(summary_output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
