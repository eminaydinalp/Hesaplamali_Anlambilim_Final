from __future__ import annotations

import argparse
import json
import os
import re
from json import JSONDecodeError
from pathlib import Path
from time import sleep, time

from dotenv import load_dotenv
from tqdm.auto import tqdm

from answer_utils import canonical_number, extract_model_final_answer, numbers_match
from evaluate_openrouter_teacher import (
    DEFAULT_MODEL,
    build_client,
    call_teacher,
)


PROMPT_TEMPLATE = """Aşağıdaki "Orijinal soru" bölümünde verilen problemin aynı matematiksel beceriyi ölçen yeni bir Türkçe versiyonunu üret.

Kurallar:
- Türkçe yaz.
- Orijinal soruyu veya cümle kalıplarını kopyalama.
- Bağlamı, isimleri, nesneleri ve tüm sayıları değiştir.
- "Orijinal doğru cevap" değerini yeni soruda kullanma.
- "Orijinal teacher çözümü" yalnızca matematiksel yapıyı anlaman içindir; çözüm adımlarını, sayıları veya cümleleri kopyalama.
- Final cevap, orijinal final cevapla aynı olmasın.
- Matematiksel yapı ve zorluk seviyesi benzer kalsın.
- Ürettiğin çözüm yalnızca yeni benzer soruyu çözsün, orijinal soruyu çözmesin.
- Çözüm 2-5 kısa hesap adımı içersin.
- similar_solution alanının son satırı tam olarak şu formatta olsun: Final cevap: <sayı>
- Final cevap yalnızca tek bir tam sayı veya ondalık sayı olsun; saat biçimi (16:40), birim, yaklaşık işareti veya açıklama kullanma.
- Cevap saat gerektiriyorsa problemi saati değil, geçen süreyi dakika veya saat cinsinden soracak şekilde yeniden kur.
- similar_answer alanı, similar_solution içindeki Final cevap değeriyle birebir aynı sayı olsun.
- Yalnızca geçerli JSON döndür. JSON dışında açıklama yazma.

JSON şeması:
{{
  "similar_question": "...",
  "similar_solution": "...",
  "similar_answer": "<sayı>"
}}

Orijinal soru:
{question}

Orijinal doğru cevap:
{reference_answer}

Orijinal teacher çözümü:
{teacher_solution}
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate similar questions for verified failed examples."
    )
    parser.add_argument("--input", default="data/train_final_500.jsonl")
    parser.add_argument("--output", default="data/similar_questions.jsonl")
    parser.add_argument(
        "--summary-output",
        default="logs/similar_questions_summary.json",
    )
    parser.add_argument("--model-id", default=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
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
    if not path.exists():
        return rows
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


def load_existing_rows(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}

    rows: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except JSONDecodeError:
                continue
            if (
                row.get("generation_error") is not None
                or row.get("processing_error") is not None
                or row.get("question") is None
                or row.get("reference_answer") is None
                or numbers_match(
                    str(row.get("reference_answer_raw")),
                    str(row.get("original_reference_answer")),
                )
            ):
                continue
            rows[str(row["original_id"])] = row
    return rows


def make_prompt(row: dict[str, object]) -> str:
    return PROMPT_TEMPLATE.format(
        question=str(row["question"]).strip(),
        reference_answer=str(row.get("reference_answer", "")).strip(),
        teacher_solution=str(row.get("teacher_model_output", "")).strip(),
    )


def extract_json_object(text: str | None) -> dict[str, object] | None:
    if not text:
        return None

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(stripped[start : end + 1])
    except JSONDecodeError:
        return extract_json_like_fields(stripped)
    return parsed if isinstance(parsed, dict) else None


def extract_json_like_fields(text: str) -> dict[str, object] | None:
    patterns = {
        "similar_question": (
            r'"similar_question"\s*:\s*"(?P<value>.*?)"\s*,\s*"similar_solution"'
        ),
        "similar_solution": (
            r'"similar_solution"\s*:\s*"(?P<value>.*?)"\s*,\s*"similar_answer"'
        ),
        "similar_answer": r'"similar_answer"\s*:\s*"(?P<value>.*?)"',
    }
    parsed: dict[str, object] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            return None
        parsed[key] = match.group("value").replace("\\n", "\n").strip()
    return parsed


def parse_similar_generation(
    model_output: str | None,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    try:
        parsed = extract_json_object(model_output)
        if parsed is None:
            return None, None, None, None, "JSONParseError: no valid JSON object found"

        question = parsed.get("similar_question")
        solution = parsed.get("similar_solution")
        answer_raw = parsed.get("similar_answer")

        question_text = str(question).strip() if question is not None else None
        solution_text = str(solution).strip() if solution is not None else None
        answer_text = str(answer_raw).strip() if answer_raw is not None else None

        if not answer_text:
            answer_text = extract_model_final_answer(solution_text)
        if re.search(r"\b\d{1,2}:\d{2}\b", answer_text or ""):
            return (
                question_text,
                solution_text,
                answer_text,
                None,
                "NonNumericAnswerError: similar_answer uses a clock-time format",
            )
        if re.search(
            r"(?im)^\s*final\s+cevap\s*:\s*\d{1,2}:\d{2}\s*$",
            solution_text or "",
        ):
            return (
                question_text,
                solution_text,
                answer_text,
                None,
                "NonNumericAnswerError: solution final answer uses a clock-time format",
            )
        answer = canonical_number(answer_text)
        solution_final_answer_raw = extract_model_final_answer(solution_text)
        solution_final_answer = canonical_number(solution_final_answer_raw)

        missing = [
            name
            for name, value in [
                ("similar_question", question_text),
                ("similar_solution", solution_text),
                ("similar_answer", answer),
                ("solution_final_answer", solution_final_answer),
            ]
            if not value
        ]
        if missing:
            return (
                question_text,
                solution_text,
                answer_text,
                answer,
                f"MissingFieldError: {', '.join(missing)}",
            )

        if not numbers_match(answer_text, solution_final_answer_raw):
            return (
                question_text,
                solution_text,
                answer_text,
                answer,
                "AnswerMismatchError: similar_answer does not match "
                "similar_solution final answer",
            )

        return question_text, solution_text, answer_text, answer, None
    except Exception as error:
        return None, None, None, None, f"{type(error).__name__}: {error}"


def build_output_row(
    source_row: dict[str, object],
    model_id: str,
    prompt: str,
    model_output: str | None,
    generation_error: str | None,
) -> dict[str, object]:
    similar_question, similar_solution, answer_raw, answer, processing_error = (
        parse_similar_generation(model_output)
        if generation_error is None
        else (None, None, None, None, None)
    )
    original_id = str(source_row["id"])

    if (
        processing_error is None
        and answer_raw is not None
        and source_row.get("reference_answer") is not None
        and numbers_match(str(answer_raw), str(source_row.get("reference_answer")))
    ):
        processing_error = (
            "AnswerReuseError: similar_answer matches original reference answer"
        )

    return {
        "id": f"{original_id}_similar",
        "original_id": original_id,
        "source_dataset": source_row.get("source_dataset"),
        "source_split": source_row.get("source_split"),
        "teacher_model_id": model_id,
        "original_question": source_row.get("question"),
        "original_reference_answer": source_row.get("reference_answer"),
        "original_teacher_solution": source_row.get("teacher_model_output"),
        "question": similar_question,
        "answer": similar_solution,
        "reference_answer_raw": answer_raw,
        "reference_answer": answer,
        "prompt": prompt,
        "model_output": model_output,
        "generation_error": generation_error,
        "processing_error": processing_error,
    }


def write_summary(
    source_rows: list[dict[str, object]],
    generated_by_original_id: dict[str, dict[str, object]],
    output_path: Path,
    summary_path: Path,
    model_id: str,
    started_at: float,
) -> None:
    ordered_rows = [
        generated_by_original_id[str(row["id"])]
        for row in source_rows
        if str(row["id"]) in generated_by_original_id
    ]
    valid_rows = [
        row
        for row in ordered_rows
        if row.get("generation_error") is None
        and row.get("processing_error") is None
    ]
    error_rows = [
        row
        for row in ordered_rows
        if row.get("generation_error") is not None
        or row.get("processing_error") is not None
    ]

    summary = {
        "teacher_model_id": model_id,
        "input_rows": len(source_rows),
        "generated_rows": len(ordered_rows),
        "valid_rows": len(valid_rows),
        "error_rows": len(error_rows),
        "output": str(output_path),
        "elapsed_seconds": round(time() - started_at, 2),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    started_at = time()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)

    source_rows = read_jsonl(input_path)
    if args.limit is not None:
        source_rows = source_rows[: args.limit]

    generated_by_original_id = (
        {} if args.restart else load_existing_rows(output_path)
    )
    pending_rows = [
        row for row in source_rows if str(row["id"]) not in generated_by_original_id
    ]

    client = build_client()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.restart else "a"

    with output_path.open(mode, encoding="utf-8") as output_file:
        progress = tqdm(pending_rows, desc="Generating similar questions")
        for index, row in enumerate(progress, start=1):
            prompt = make_prompt(row)
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
            output_row = build_output_row(
                source_row=row,
                model_id=args.model_id,
                prompt=prompt,
                model_output=model_output,
                generation_error=generation_error,
            )
            generated_by_original_id[str(row["id"])] = output_row
            output_file.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            output_file.flush()

            if index % args.checkpoint_every == 0:
                write_summary(
                    source_rows=source_rows,
                    generated_by_original_id=generated_by_original_id,
                    output_path=output_path,
                    summary_path=summary_path,
                    model_id=args.model_id,
                    started_at=started_at,
                )

            if args.sleep_seconds > 0:
                sleep(args.sleep_seconds)

    write_summary(
        source_rows=source_rows,
        generated_by_original_id=generated_by_original_id,
        output_path=output_path,
        summary_path=summary_path,
        model_id=args.model_id,
        started_at=started_at,
    )
    write_jsonl(
        output_path,
        [
            generated_by_original_id[str(row["id"])]
            for row in source_rows
            if str(row["id"]) in generated_by_original_id
        ],
    )
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
