from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import load_dataset


DATASET_ID = os.getenv("GSM8K_TR_DATASET_ID", "ytu-ce-cosmos/gsm8k_tr")
OUTPUT_PATH = Path(os.getenv("GSM8K_TR_OUTPUT_PATH", "data/gsm8k_tr.jsonl"))
INFO_PATH = Path("logs/gsm8k_tr_dataset_info.json")


def normalize_example(example: dict[str, str], index: int, split: str) -> dict[str, str]:
    answer = example["answer"].strip()
    return {
        "id": f"{split}-{index}",
        "source_dataset": DATASET_ID,
        "source_split": split,
        "question": example["question"].strip(),
        "answer": answer,
    }


def main() -> None:
    dataset = load_dataset(DATASET_ID)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    INFO_PATH.parent.mkdir(parents=True, exist_ok=True)

    split_counts = {split: len(dataset[split]) for split in dataset}
    columns = {split: dataset[split].column_names for split in dataset}

    total_rows = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for split in dataset:
            for index, example in enumerate(dataset[split]):
                row = normalize_example(example, index=index, split=split)
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += 1

    info = {
        "dataset_id": DATASET_ID,
        "output_path": str(OUTPUT_PATH),
        "split_counts": split_counts,
        "columns": columns,
        "total_rows": total_rows,
        "standardized_columns": [
            "id",
            "source_dataset",
            "source_split",
            "question",
            "answer",
        ],
    }
    INFO_PATH.write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
