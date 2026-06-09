from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from json import JSONDecodeError
from pathlib import Path
from time import time
from typing import Any

import torch
from tqdm.auto import tqdm

from evaluate_m1_failures import generate_batch_with_fallback, make_prompt
from evaluate_qwen_test_baseline import (
    merge_test_rows,
    parse_prediction,
    prediction_matches_current_row,
)
from selective_loop import (
    DEFAULT_MODEL_ID,
    adapter_checkpoint_exists,
    load_model_and_tokenizer,
    read_jsonl,
)


STRATEGY_DEFAULTS = {
    "selective": {
        "adapter_dir": "models/selective_loop/active_adapter",
        "predictions_output": "data/test_selective_predictions.jsonl",
        "failed_output": "data/test_selective_failed.jsonl",
        "summary_output": "logs/selective_test_summary.json",
        "loop_summary": "logs/selective_loop_summary.json",
    },
    "blind": {
        "adapter_dir": "models/blind_loop/active_adapter",
        "predictions_output": "data/test_blind_predictions.jsonl",
        "failed_output": "data/test_blind_failed.jsonl",
        "summary_output": "logs/blind_test_summary.json",
        "loop_summary": "logs/blind_loop_summary.json",
    },
}


EVALUATION_FIELDS = [
    "model_name",
    "strategy",
    "model_id",
    "adapter_dir",
    "total_rows",
    "evaluated_rows",
    "scored_rows",
    "correct_rows",
    "failed_rows",
    "error_rows",
    "accuracy",
    "m1_failed_test_total",
    "m1_failed_test_correct",
    "m1_failed_test_accuracy",
    "test_generalization_score",
    "accuracy_gain_vs_m1",
    "similar_correct_rows",
    "similar_accuracy",
    "predictions_output",
    "failed_output",
    "summary_output",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate final selective/blind LoRA adapters on the fixed test set."
    )
    parser.add_argument("--model-id", default=os.getenv("M1_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--test-input", default="data/test.jsonl")
    parser.add_argument(
        "--references-input",
        default="data/test_with_reference_numeric.jsonl",
    )
    parser.add_argument(
        "--strategy",
        choices=["selective", "blind", "all", "aggregate"],
        default="all",
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Adapter path for single-strategy runs. Defaults to the known Phase 4/5 adapter.",
    )
    parser.add_argument(
        "--predictions-output",
        default=None,
        help="Prediction JSONL path for single-strategy runs.",
    )
    parser.add_argument(
        "--failed-output",
        default=None,
        help="Failed prediction JSONL path for single-strategy runs.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Summary JSON path for single-strategy runs.",
    )
    parser.add_argument(
        "--evaluation-output",
        default="results/evaluation.csv",
        help="CSV comparison output.",
    )
    parser.add_argument(
        "--baseline-summary",
        default="logs/qwen_test_baseline_summary.json",
    )
    parser.add_argument(
        "--baseline-failed",
        default="data/test_qwen_failed.jsonl",
    )
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    predictions: dict[str, dict[str, Any]] = {}
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


def prediction_matches_run(
    prediction: dict[str, Any],
    row: dict[str, Any],
    strategy: str,
    adapter_dir: Path,
    model_id: str,
) -> bool:
    return (
        prediction_matches_current_row(prediction, row)
        and str(prediction.get("strategy")) == strategy
        and str(prediction.get("adapter_dir")) == str(adapter_dir)
        and str(prediction.get("model_id")) == model_id
    )


def drop_stale_adapter_predictions(
    predictions_by_id: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    strategy: str,
    adapter_dir: Path,
    model_id: str,
) -> int:
    rows_by_id = {str(row["id"]): row for row in rows}
    stale_ids = [
        row_id
        for row_id, prediction in predictions_by_id.items()
        if row_id not in rows_by_id
        or not prediction_matches_run(
            prediction=prediction,
            row=rows_by_id[row_id],
            strategy=strategy,
            adapter_dir=adapter_dir,
            model_id=model_id,
        )
    ]
    for row_id in stale_ids:
        del predictions_by_id[row_id]
    return len(stale_ids)


def strategy_paths(
    args: argparse.Namespace,
    strategy: str,
) -> tuple[Path, Path, Path, Path, Path]:
    defaults = STRATEGY_DEFAULTS[strategy]
    adapter_dir = Path(args.adapter_dir or defaults["adapter_dir"])
    predictions_output = Path(args.predictions_output or defaults["predictions_output"])
    failed_output = Path(args.failed_output or defaults["failed_output"])
    summary_output = Path(args.summary_output or defaults["summary_output"])
    loop_summary = Path(defaults["loop_summary"])
    return adapter_dir, predictions_output, failed_output, summary_output, loop_summary


def load_test_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    test_rows = read_jsonl(Path(args.test_input))
    reference_rows = read_jsonl(Path(args.references_input))
    rows = merge_test_rows(test_rows, reference_rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def load_m1_failed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["id"]) for row in read_jsonl(path)}


def summarize_predictions(
    rows: list[dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
    m1_failed_ids: set[str],
    predictions_output: Path,
    failed_output: Path,
    summary_output: Path,
    strategy: str,
    adapter_dir: Path,
    model_id: str,
    device: str,
    started_at: float,
    stale_existing_predictions: int,
) -> dict[str, Any]:
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
    failed_rows = [row for row in scored_rows if row.get("is_correct") is False]
    correct_rows = [row for row in scored_rows if row.get("is_correct") is True]
    error_rows = [
        row
        for row in ordered_predictions
        if row.get("generation_error") is not None
        or row.get("processing_error") is not None
    ]
    model_parse_failed_rows = [
        row for row in scored_rows if row.get("predicted_answer_raw") is None
    ]
    m1_failed_scored_rows = [
        row for row in scored_rows if str(row.get("id")) in m1_failed_ids
    ]
    m1_failed_correct_rows = [
        row for row in m1_failed_scored_rows if row.get("is_correct") is True
    ]

    write_jsonl(failed_output, failed_rows)
    summary = {
        "model_id": model_id,
        "strategy": strategy,
        "adapter_dir": str(adapter_dir),
        "device": device,
        "total_input_rows": len(rows),
        "evaluated_rows": len(ordered_predictions),
        "scored_rows": len(scored_rows),
        "correct_rows": len(correct_rows),
        "failed_rows": len(failed_rows),
        "error_rows": len(error_rows),
        "model_parse_failed_rows": len(model_parse_failed_rows),
        "m1_failed_test_total": len(m1_failed_scored_rows),
        "m1_failed_test_correct": len(m1_failed_correct_rows),
        "m1_failed_test_accuracy": (
            len(m1_failed_correct_rows) / len(m1_failed_scored_rows)
            if m1_failed_scored_rows
            else None
        ),
        "stale_existing_predictions": stale_existing_predictions,
        "accuracy": len(correct_rows) / len(scored_rows) if scored_rows else None,
        "predictions_output": str(predictions_output),
        "failed_output": str(failed_output),
        "elapsed_seconds": round(time() - started_at, 2),
    }
    write_json(summary_output, summary)
    return summary


def evaluate_strategy(args: argparse.Namespace, strategy: str) -> dict[str, Any]:
    started_at = time()
    rows = load_test_rows(args)
    adapter_dir, predictions_output, failed_output, summary_output, _ = strategy_paths(
        args=args,
        strategy=strategy,
    )
    if not adapter_checkpoint_exists(adapter_dir):
        raise FileNotFoundError(f"No adapter checkpoint found at {adapter_dir}")

    predictions_by_id = (
        {} if args.restart else load_existing_predictions(predictions_output)
    )
    stale_existing_predictions = drop_stale_adapter_predictions(
        predictions_by_id=predictions_by_id,
        rows=rows,
        strategy=strategy,
        adapter_dir=adapter_dir,
        model_id=args.model_id,
    )
    pending_rows = [
        row for row in rows if str(row["id"]) not in predictions_by_id
    ]

    tokenizer, model, device = load_model_and_tokenizer(args, adapter_dir)
    model.eval()
    m1_failed_ids = load_m1_failed_ids(Path(args.baseline_failed))
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.restart else "a"
    with predictions_output.open(mode, encoding="utf-8") as output_file:
        progress = tqdm(
            range(0, len(pending_rows), args.batch_size),
            desc=f"{strategy} test eval",
        )
        next_checkpoint = args.checkpoint_every
        processed_rows = 0
        for start in progress:
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
                    "model_type": "qwen3_5_lora_adapter",
                    "strategy": strategy,
                    "adapter_dir": str(adapter_dir),
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
            processed_rows += len(batch_rows)

            if processed_rows >= next_checkpoint:
                summarize_predictions(
                    rows=rows,
                    predictions_by_id=predictions_by_id,
                    m1_failed_ids=m1_failed_ids,
                    predictions_output=predictions_output,
                    failed_output=failed_output,
                    summary_output=summary_output,
                    strategy=strategy,
                    adapter_dir=adapter_dir,
                    model_id=args.model_id,
                    device=device,
                    started_at=started_at,
                    stale_existing_predictions=stale_existing_predictions,
                )
                next_checkpoint += args.checkpoint_every

    summary = summarize_predictions(
        rows=rows,
        predictions_by_id=predictions_by_id,
        m1_failed_ids=m1_failed_ids,
        predictions_output=predictions_output,
        failed_output=failed_output,
        summary_output=summary_output,
        strategy=strategy,
        adapter_dir=adapter_dir,
        model_id=args.model_id,
        device=device,
        started_at=started_at,
        stale_existing_predictions=stale_existing_predictions,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def csv_number(value: Any) -> Any:
    if value is None:
        return ""
    return value


def add_baseline_row(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> float | None:
    baseline_summary = read_json(Path(args.baseline_summary))
    if not baseline_summary:
        return None

    accuracy = baseline_summary.get("accuracy")
    m1_failed_total = baseline_summary.get("failed_rows")
    rows.append(
        {
            "model_name": "M1 baseline",
            "strategy": "baseline",
            "model_id": baseline_summary.get("model_id", args.model_id),
            "adapter_dir": "",
            "total_rows": baseline_summary.get("total_input_rows"),
            "evaluated_rows": baseline_summary.get("evaluated_rows"),
            "scored_rows": baseline_summary.get("scored_rows"),
            "correct_rows": baseline_summary.get("correct_rows"),
            "failed_rows": baseline_summary.get("failed_rows"),
            "error_rows": baseline_summary.get("error_rows"),
            "accuracy": accuracy,
            "m1_failed_test_total": m1_failed_total,
            "m1_failed_test_correct": 0 if m1_failed_total is not None else "",
            "m1_failed_test_accuracy": 0 if m1_failed_total else "",
            "test_generalization_score": accuracy,
            "accuracy_gain_vs_m1": 0 if accuracy is not None else "",
            "similar_correct_rows": "",
            "similar_accuracy": "",
            "predictions_output": baseline_summary.get("predictions_output"),
            "failed_output": baseline_summary.get("failed_output"),
            "summary_output": args.baseline_summary,
        }
    )
    return float(accuracy) if accuracy is not None else None


def add_strategy_row(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    strategy: str,
    baseline_accuracy: float | None,
) -> None:
    adapter_dir, _, _, summary_output, loop_summary_path = strategy_paths(args, strategy)
    summary = read_json(summary_output)
    if not summary:
        return

    loop_summary = read_json(loop_summary_path)
    similar_correct_rows = (
        loop_summary.get("accepted_rows")
        if strategy == "selective"
        else loop_summary.get("similar_correct_rows")
    )
    similar_total = loop_summary.get("total_aligned_examples")
    similar_accuracy = (
        similar_correct_rows / similar_total
        if similar_correct_rows is not None and similar_total
        else None
    )
    accuracy = summary.get("accuracy")
    rows.append(
        {
            "model_name": f"{strategy} final adapter",
            "strategy": strategy,
            "model_id": summary.get("model_id", args.model_id),
            "adapter_dir": summary.get("adapter_dir", str(adapter_dir)),
            "total_rows": summary.get("total_input_rows"),
            "evaluated_rows": summary.get("evaluated_rows"),
            "scored_rows": summary.get("scored_rows"),
            "correct_rows": summary.get("correct_rows"),
            "failed_rows": summary.get("failed_rows"),
            "error_rows": summary.get("error_rows"),
            "accuracy": accuracy,
            "m1_failed_test_total": summary.get("m1_failed_test_total"),
            "m1_failed_test_correct": summary.get("m1_failed_test_correct"),
            "m1_failed_test_accuracy": summary.get("m1_failed_test_accuracy"),
            "test_generalization_score": accuracy,
            "accuracy_gain_vs_m1": (
                accuracy - baseline_accuracy
                if accuracy is not None and baseline_accuracy is not None
                else None
            ),
            "similar_correct_rows": similar_correct_rows,
            "similar_accuracy": similar_accuracy,
            "predictions_output": summary.get("predictions_output"),
            "failed_output": summary.get("failed_output"),
            "summary_output": str(summary_output),
        }
    )


def csv_strategies(args: argparse.Namespace) -> list[str]:
    if args.strategy in {"all", "aggregate"}:
        return ["selective", "blind"]
    return [args.strategy]


def write_evaluation_csv(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_accuracy = add_baseline_row(rows, args)
    for strategy in csv_strategies(args):
        add_strategy_row(rows, args, strategy, baseline_accuracy)

    output_path = Path(args.evaluation_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_number(row.get(field)) for field in EVALUATION_FIELDS})
    return rows


def main() -> None:
    args = parse_args()
    strategies = (
        ["selective", "blind"]
        if args.strategy == "all"
        else [] if args.strategy == "aggregate" else [args.strategy]
    )
    if args.strategy == "all" and any(
        option is not None
        for option in (args.adapter_dir, args.predictions_output, args.failed_output, args.summary_output)
    ):
        raise ValueError(
            "--adapter-dir/--predictions-output/--failed-output/--summary-output "
            "can only be used with a single strategy."
        )

    for strategy in strategies:
        evaluate_strategy(args, strategy)

    evaluation_rows = write_evaluation_csv(args)
    print(json.dumps(evaluation_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
