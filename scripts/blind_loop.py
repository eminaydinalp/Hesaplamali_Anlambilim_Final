from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from time import time

from tqdm.auto import tqdm
from transformers import set_seed

from selective_loop import (
    DEFAULT_MODEL_ID,
    adapter_checkpoint_exists,
    build_examples,
    commit_step,
    evaluate_candidate,
    finalize_pending_commit,
    load_log_rows,
    load_model_and_tokenizer,
    read_jsonl,
    save_adapter,
    summarize_log_rows,
    train_one_example,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the blind LoRA fine-tuning loop.")
    parser.add_argument("--model-id", default=os.getenv("M1_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--verified-input", default="data/train_final_500.jsonl")
    parser.add_argument("--solutions-input", default="data/solutions_final_500.jsonl")
    parser.add_argument("--similar-input", default="data/similar_questions.jsonl")
    parser.add_argument("--output-dir", default="models/blind_loop")
    parser.add_argument(
        "--active-adapter-dir",
        default=None,
        help="Defaults to <output-dir>/active_adapter.",
    )
    parser.add_argument(
        "--candidate-adapter-dir",
        default=None,
        help="Defaults to <output-dir>/candidate_adapter_tmp.",
    )
    parser.add_argument("--log-output", default="logs/blind_loop_log.csv")
    parser.add_argument("--state-output", default="logs/blind_loop_state.json")
    parser.add_argument("--summary-output", default="logs/blind_loop_summary.json")
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-train-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--train-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_summary(
    args: argparse.Namespace,
    examples,
    log_rows: list[dict[str, str]],
    summary_path: Path,
    active_adapter_dir: Path,
    candidate_adapter_dir: Path,
    started_at: float,
    dry_run: bool,
) -> None:
    counts = summarize_log_rows(log_rows)
    correct_rows = sum(row.get("is_correct", "").lower() == "true" for row in log_rows)
    summary = {
        "model_id": args.model_id,
        "strategy": "blind",
        "dry_run": dry_run,
        "train_epochs": args.train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "max_train_tokens": args.max_train_tokens,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "seed": args.seed,
        "total_aligned_examples": len(examples),
        "pending_rows": len(examples) - counts["processed_rows"],
        "active_adapter_dir": str(active_adapter_dir),
        "candidate_adapter_dir": str(candidate_adapter_dir),
        "log_output": args.log_output,
        "state_output": args.state_output,
        "elapsed_seconds": round(time() - started_at, 2),
        "similar_correct_rows": correct_rows,
        "similar_wrong_rows": len(log_rows) - correct_rows,
        **counts,
    }
    write_json(summary_path, summary)


def main() -> None:
    args = parse_args()
    started_at = time()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    active_adapter_dir = Path(args.active_adapter_dir or output_dir / "active_adapter")
    candidate_adapter_dir = Path(
        args.candidate_adapter_dir or output_dir / "candidate_adapter_tmp"
    )
    log_path = Path(args.log_output)
    state_path = Path(args.state_output)
    summary_path = Path(args.summary_output)

    verified_rows = read_jsonl(Path(args.verified_input))
    solution_rows = read_jsonl(Path(args.solutions_input))
    similar_rows = read_jsonl(Path(args.similar_input))
    examples = build_examples(
        verified_rows=verified_rows,
        solution_rows=solution_rows,
        similar_rows=similar_rows,
        limit=args.limit,
    )

    if (
        args.limit is None
        and not args.allow_incomplete
        and len(examples) < args.min_rows
    ):
        raise ValueError(
            f"Only {len(examples)} aligned examples are available; "
            f"--min-rows is {args.min_rows}."
        )

    if args.restart:
        if log_path.exists():
            log_path.unlink()
        if state_path.exists():
            state_path.unlink()
        if summary_path.exists():
            summary_path.unlink()
        if active_adapter_dir.exists():
            shutil.rmtree(active_adapter_dir)
        if candidate_adapter_dir.exists():
            shutil.rmtree(candidate_adapter_dir)

    finalize_pending_commit(
        args=args,
        examples=examples,
        active_adapter_dir=active_adapter_dir,
        candidate_adapter_dir=candidate_adapter_dir,
        log_path=log_path,
        state_path=state_path,
    )

    log_rows = load_log_rows(log_path)
    completed_ids = {row.get("original_id") for row in log_rows}
    pending_examples = [
        example for example in examples if example.original_id not in completed_ids
    ]

    if args.dry_run:
        write_summary(
            args=args,
            examples=examples,
            log_rows=log_rows,
            summary_path=summary_path,
            active_adapter_dir=active_adapter_dir,
            candidate_adapter_dir=candidate_adapter_dir,
            started_at=started_at,
            dry_run=True,
        )
        print(summary_path.read_text(encoding="utf-8"))
        return

    tokenizer, model, device = load_model_and_tokenizer(args, active_adapter_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not adapter_checkpoint_exists(active_adapter_dir):
        save_adapter(model, active_adapter_dir)

    counters = summarize_log_rows(log_rows)
    progress = tqdm(pending_examples, desc="Blind LoRA loop")
    for example in progress:
        step_started_at = time()
        accepted_before = counters["accepted_rows"]
        train_loss: float | None = None
        train_last_loss: float | None = None

        try:
            train_loss, train_last_loss = train_one_example(
                model=model,
                tokenizer=tokenizer,
                example=example,
                args=args,
            )
            save_adapter(model, candidate_adapter_dir)
            evaluation = evaluate_candidate(
                model=model,
                tokenizer=tokenizer,
                example=example,
                args=args,
            )
        except Exception as error:
            evaluation = {
                "model_output": None,
                "predicted_answer_raw": None,
                "predicted_answer": None,
                "is_correct": False,
                "generation_error": None,
                "processing_error": f"{type(error).__name__}: {error}",
            }

        # Blind strategy: keep every update, even when qii is wrong.
        save_adapter(model, candidate_adapter_dir)
        accepted = True
        log_row = {
            "step_index": example.step_index,
            "original_id": example.original_id,
            "similar_id": example.similar_id,
            "accepted_before": accepted_before,
            "accepted_after": accepted_before + 1,
            "active_adapter_dir": str(active_adapter_dir),
            "candidate_adapter_dir": str(candidate_adapter_dir),
            "train_loss": train_loss,
            "train_last_loss": train_last_loss,
            "similar_reference_answer_raw": example.similar_reference_answer_raw,
            "similar_reference_answer": example.similar_reference_answer,
            "predicted_answer_raw": evaluation.get("predicted_answer_raw"),
            "predicted_answer": evaluation.get("predicted_answer"),
            "is_correct": evaluation.get("is_correct"),
            "accepted": accepted,
            "model_output": evaluation.get("model_output"),
            "generation_error": evaluation.get("generation_error"),
            "processing_error": evaluation.get("processing_error"),
            "elapsed_seconds": round(time() - step_started_at, 2),
        }
        counters = commit_step(
            args=args,
            examples=examples,
            log_row=log_row,
            active_adapter_dir=active_adapter_dir,
            candidate_adapter_dir=candidate_adapter_dir,
            log_path=log_path,
            state_path=state_path,
            counters_before=counters,
        )
        progress.set_postfix(
            kept=counters["accepted_rows"],
            similar_correct=evaluation.get("is_correct"),
            device=device,
        )

    final_log_rows = load_log_rows(log_path)
    write_summary(
        args=args,
        examples=examples,
        log_rows=final_log_rows,
        summary_path=summary_path,
        active_adapter_dir=active_adapter_dir,
        candidate_adapter_dir=candidate_adapter_dir,
        started_at=started_at,
        dry_run=False,
    )
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
