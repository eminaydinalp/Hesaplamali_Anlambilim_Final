from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from statistics import mean
from time import time
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from answer_utils import canonical_number, extract_model_final_answer, numbers_match
from evaluate_m1_failures import choose_device, make_prompt


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"
LOG_FIELDS = [
    "step_index",
    "original_id",
    "similar_id",
    "accepted_before",
    "accepted_after",
    "active_adapter_dir",
    "candidate_adapter_dir",
    "train_loss",
    "train_last_loss",
    "similar_reference_answer_raw",
    "similar_reference_answer",
    "predicted_answer_raw",
    "predicted_answer",
    "is_correct",
    "accepted",
    "model_output",
    "generation_error",
    "processing_error",
    "elapsed_seconds",
]


@dataclass(frozen=True)
class LoopExample:
    step_index: int
    original_id: str
    similar_id: str
    train_question: str
    train_solution: str
    similar_question: str
    similar_reference_answer_raw: str
    similar_reference_answer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the selective LoRA fine-tuning loop."
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("M1_MODEL_ID", DEFAULT_MODEL_ID),
    )
    parser.add_argument(
        "--verified-input",
        default="data/train_final_500.jsonl",
    )
    parser.add_argument("--solutions-input", default="data/solutions_final_500.jsonl")
    parser.add_argument("--similar-input", default="data/similar_questions.jsonl")
    parser.add_argument("--output-dir", default="models/selective_loop")
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
    parser.add_argument("--log-output", default="logs/loop_log.csv")
    parser.add_argument("--state-output", default="logs/selective_loop_state.json")
    parser.add_argument("--summary-output", default="logs/selective_loop_summary.json")
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow fewer than --min-rows aligned examples when --limit is not set.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and resume state without loading or training the model.",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-train-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=160)
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


def parse_lora_targets(value: str) -> list[str]:
    targets = [target.strip() for target in value.split(",")]
    return [target for target in targets if target]


def adapter_checkpoint_exists(path: Path) -> bool:
    return (path / "adapter_config.json").exists()


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_directory(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Cannot copy missing directory: {source}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def build_examples(
    verified_rows: list[dict[str, Any]],
    solution_rows: list[dict[str, Any]],
    similar_rows: list[dict[str, Any]],
    limit: int | None,
) -> list[LoopExample]:
    selected_verified_rows = verified_rows[:limit] if limit is not None else verified_rows
    solutions_by_id = {str(row.get("id")): row for row in solution_rows}
    similar_by_original_id = {
        str(row.get("original_id")): row for row in similar_rows
        if row.get("original_id") is not None
    }

    examples: list[LoopExample] = []
    missing: list[str] = []
    invalid: list[str] = []

    for step_index, verified_row in enumerate(selected_verified_rows, start=1):
        original_id = str(verified_row.get("id"))
        solution_row = solutions_by_id.get(original_id)
        similar_row = similar_by_original_id.get(original_id)

        if solution_row is None or similar_row is None:
            missing_parts = []
            if solution_row is None:
                missing_parts.append("solution")
            if similar_row is None:
                missing_parts.append("similar")
            missing.append(f"{original_id} ({', '.join(missing_parts)})")
            continue

        if (
            similar_row.get("generation_error") is not None
            or similar_row.get("processing_error") is not None
        ):
            invalid.append(f"{original_id} (similar generation error)")
            continue

        train_question = str(verified_row.get("question", "")).strip()
        train_solution = str(
            solution_row.get("solution")
            or verified_row.get("teacher_model_output")
            or ""
        ).strip()
        similar_question = str(similar_row.get("question", "")).strip()
        similar_reference_answer_raw = str(
            similar_row.get("reference_answer_raw")
            or similar_row.get("reference_answer")
            or ""
        ).strip()
        similar_reference_answer = canonical_number(similar_reference_answer_raw)

        required_values = {
            "train_question": train_question,
            "train_solution": train_solution,
            "similar_question": similar_question,
            "similar_reference_answer_raw": similar_reference_answer_raw,
            "similar_reference_answer": similar_reference_answer,
        }
        empty_fields = [
            name for name, value in required_values.items() if not value
        ]
        if empty_fields:
            invalid.append(f"{original_id} ({', '.join(empty_fields)})")
            continue

        examples.append(
            LoopExample(
                step_index=step_index,
                original_id=original_id,
                similar_id=str(similar_row.get("id") or f"{original_id}_similar"),
                train_question=train_question,
                train_solution=train_solution,
                similar_question=similar_question,
                similar_reference_answer_raw=similar_reference_answer_raw,
                similar_reference_answer=str(similar_reference_answer),
            )
        )

    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} verified rows are missing aligned inputs. "
            f"Examples: {sample}"
        )
    if invalid:
        sample = ", ".join(invalid[:5])
        raise ValueError(
            f"{len(invalid)} aligned rows are invalid. Examples: {sample}"
        )

    return examples


def load_log_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def bool_from_csv(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def summarize_log_rows(rows: list[dict[str, str]]) -> dict[str, int]:
    processed = len(rows)
    accepted = sum(1 for row in rows if bool_from_csv(row.get("accepted")))
    rejected = processed - accepted
    return {
        "processed_rows": processed,
        "accepted_rows": accepted,
        "rejected_rows": rejected,
    }


def append_log_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_header = path.exists() and path.stat().st_size > 0
    normalized_row = {field: row.get(field, "") for field in LOG_FIELDS}
    with path.open("a", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=LOG_FIELDS)
        if not has_header:
            writer.writeheader()
        writer.writerow(normalized_row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_state_payload(
    args: argparse.Namespace,
    examples: list[LoopExample],
    log_summary: dict[str, int],
    active_adapter_dir: Path,
    candidate_adapter_dir: Path,
    pending_commit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "updated_at": utc_now(),
        "model_id": args.model_id,
        "total_aligned_examples": len(examples),
        "limit": args.limit,
        "min_rows": args.min_rows,
        "active_adapter_dir": str(active_adapter_dir),
        "candidate_adapter_dir": str(candidate_adapter_dir),
        "log_output": args.log_output,
        "pending_commit": pending_commit,
        **log_summary,
    }


def finalize_pending_commit(
    args: argparse.Namespace,
    examples: list[LoopExample],
    active_adapter_dir: Path,
    candidate_adapter_dir: Path,
    log_path: Path,
    state_path: Path,
) -> None:
    if not state_path.exists():
        return

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except JSONDecodeError:
        return

    pending = state.get("pending_commit")
    if not isinstance(pending, dict):
        return

    log_row = pending.get("log_row")
    if not isinstance(log_row, dict):
        return

    logged_ids = {row.get("original_id") for row in load_log_rows(log_path)}
    original_id = str(log_row.get("original_id"))
    if original_id not in logged_ids:
        if bool_from_csv(log_row.get("accepted")):
            copy_directory(candidate_adapter_dir, active_adapter_dir)
        append_log_row(log_path, log_row)

    rows = load_log_rows(log_path)
    summary = summarize_log_rows(rows)
    payload = build_state_payload(
        args=args,
        examples=examples,
        log_summary=summary,
        active_adapter_dir=active_adapter_dir,
        candidate_adapter_dir=candidate_adapter_dir,
        pending_commit=None,
    )
    write_json(state_path, payload)


def commit_step(
    args: argparse.Namespace,
    examples: list[LoopExample],
    log_row: dict[str, Any],
    active_adapter_dir: Path,
    candidate_adapter_dir: Path,
    log_path: Path,
    state_path: Path,
    counters_before: dict[str, int],
) -> dict[str, int]:
    pending_payload = build_state_payload(
        args=args,
        examples=examples,
        log_summary=counters_before,
        active_adapter_dir=active_adapter_dir,
        candidate_adapter_dir=candidate_adapter_dir,
        pending_commit={"log_row": log_row},
    )
    write_json(state_path, pending_payload)

    if bool_from_csv(log_row.get("accepted")):
        copy_directory(candidate_adapter_dir, active_adapter_dir)

    append_log_row(log_path, log_row)

    counters_after = {
        "processed_rows": counters_before["processed_rows"] + 1,
        "accepted_rows": counters_before["accepted_rows"]
        + int(bool_from_csv(log_row.get("accepted"))),
        "rejected_rows": counters_before["rejected_rows"]
        + int(not bool_from_csv(log_row.get("accepted"))),
    }
    completed_payload = build_state_payload(
        args=args,
        examples=examples,
        log_summary=counters_after,
        active_adapter_dir=active_adapter_dir,
        candidate_adapter_dir=candidate_adapter_dir,
        pending_commit=None,
    )
    write_json(state_path, completed_payload)
    return counters_after


def require_peft():
    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    except ImportError as error:
        raise RuntimeError(
            "PEFT is required for selective LoRA fine-tuning. "
            "Install it with: python -m pip install peft"
        ) from error
    return LoraConfig, PeftModel, TaskType, get_peft_model


def load_model_and_tokenizer(
    args: argparse.Namespace,
    active_adapter_dir: Path,
):
    LoraConfig, PeftModel, TaskType, get_peft_model = require_peft()

    local_files_only = not args.allow_download
    device = choose_device()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "torch_dtype": "auto",
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        **model_kwargs,
    )
    if device != "cuda":
        base_model.to(device)

    if adapter_checkpoint_exists(active_adapter_dir):
        model = PeftModel.from_pretrained(
            base_model,
            active_adapter_dir,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=parse_lora_targets(args.lora_target_modules),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, lora_config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if hasattr(model, "config"):
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id

    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype in {
            torch.float16,
            torch.bfloat16,
        }:
            parameter.data = parameter.data.float()

    model.train()
    return tokenizer, model, device


def first_parameter_device(model) -> torch.device:
    return next(model.parameters()).device


def clone_trainable_state(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_state(
    model,
    state: dict[str, torch.Tensor],
) -> None:
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        parameter = parameters[name]
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def save_adapter(model, path: Path) -> None:
    reset_directory(path)
    model.save_pretrained(path)


def build_training_tensors(
    tokenizer,
    question: str,
    solution: str,
    max_train_tokens: int,
) -> dict[str, torch.Tensor]:
    prompt_text = make_prompt(question).strip() + "\n\n"
    completion_text = solution.strip()
    eos_text = tokenizer.eos_token or ""
    if eos_text and not completion_text.endswith(eos_text):
        completion_text += eos_text

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        prompt_text + completion_text,
        add_special_tokens=False,
    )["input_ids"]

    if len(full_ids) > max_train_tokens:
        full_ids = full_ids[:max_train_tokens]

    prompt_length = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_length + full_ids[prompt_length:]
    if not any(label != -100 for label in labels):
        raise ValueError("Training example was truncated before the solution tokens.")

    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


def train_one_example(
    model,
    tokenizer,
    example: LoopExample,
    args: argparse.Namespace,
) -> tuple[float, float]:
    model.train()
    tensors = build_training_tensors(
        tokenizer=tokenizer,
        question=example.train_question,
        solution=example.train_solution,
        max_train_tokens=args.max_train_tokens,
    )
    device = first_parameter_device(model)
    tensors = {key: value.to(device) for key, value in tensors.items()}

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    losses: list[float] = []
    for _ in range(args.train_epochs):
        optimizer.zero_grad(set_to_none=True)
        output = model(**tensors)
        loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {loss.item()}")
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.max_grad_norm,
            )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return mean(losses), losses[-1]


def generate_answer(
    model,
    tokenizer,
    question: str,
    args: argparse.Namespace,
) -> tuple[str | None, str | None]:
    prompt = make_prompt(question)
    device = first_parameter_device(model)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    try:
        model.eval()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[:, input_length:]
        output_text = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        return output_text.strip(), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"
    finally:
        model.train()


def evaluate_candidate(
    model,
    tokenizer,
    example: LoopExample,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_output, generation_error = generate_answer(
        model=model,
        tokenizer=tokenizer,
        question=example.similar_question,
        args=args,
    )
    if generation_error is not None:
        return {
            "model_output": model_output,
            "predicted_answer_raw": None,
            "predicted_answer": None,
            "is_correct": False,
            "generation_error": generation_error,
            "processing_error": None,
        }

    try:
        predicted_answer_raw = extract_model_final_answer(model_output)
        predicted_answer = canonical_number(predicted_answer_raw)
        is_correct = numbers_match(
            predicted_answer_raw,
            example.similar_reference_answer_raw,
        )
        return {
            "model_output": model_output,
            "predicted_answer_raw": predicted_answer_raw,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "generation_error": None,
            "processing_error": None,
        }
    except Exception as error:
        return {
            "model_output": model_output,
            "predicted_answer_raw": None,
            "predicted_answer": None,
            "is_correct": False,
            "generation_error": None,
            "processing_error": f"{type(error).__name__}: {error}",
        }


def write_summary(
    args: argparse.Namespace,
    examples: list[LoopExample],
    log_rows: list[dict[str, str]],
    summary_path: Path,
    active_adapter_dir: Path,
    candidate_adapter_dir: Path,
    started_at: float,
    dry_run: bool,
) -> None:
    counts = summarize_log_rows(log_rows)
    summary = {
        "model_id": args.model_id,
        "dry_run": dry_run,
        "total_aligned_examples": len(examples),
        "pending_rows": len(examples) - counts["processed_rows"],
        "active_adapter_dir": str(active_adapter_dir),
        "candidate_adapter_dir": str(candidate_adapter_dir),
        "log_output": args.log_output,
        "state_output": args.state_output,
        "elapsed_seconds": round(time() - started_at, 2),
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
            f"--min-rows is {args.min_rows}. Complete Phase 3 or pass "
            "--allow-incomplete for a development run."
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
    progress = tqdm(pending_examples, desc="Selective LoRA loop")
    for example in progress:
        step_started_at = time()
        active_state = clone_trainable_state(model)
        accepted_before = counters["accepted_rows"]

        train_loss: float | None = None
        train_last_loss: float | None = None
        generation_error: str | None = None
        processing_error: str | None = None
        evaluation: dict[str, Any]

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

        accepted = bool(evaluation["is_correct"])
        if not accepted:
            restore_trainable_state(model, active_state)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if evaluation.get("generation_error") is not None:
            generation_error = str(evaluation["generation_error"])
        if evaluation.get("processing_error") is not None:
            processing_error = str(evaluation["processing_error"])

        log_row = {
            "step_index": example.step_index,
            "original_id": example.original_id,
            "similar_id": example.similar_id,
            "accepted_before": accepted_before,
            "accepted_after": accepted_before + int(accepted),
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
            "generation_error": generation_error,
            "processing_error": processing_error,
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
            accepted=counters["accepted_rows"],
            rejected=counters["rejected_rows"],
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
