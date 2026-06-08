from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer


MODEL_ID = os.getenv("M1_MODEL_ID", "Qwen/Qwen3.5-4B")
PROMPT = "Soru: 2 kalem 3 TL ise 5 kalem kac TL eder?\nCevap:"


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    device = choose_device()
    config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto" if device == "cuda" else None,
        local_files_only=True,
    )
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt")
    first_device = next(model.parameters()).device
    inputs = {key: value.to(first_device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    info = {
        "model_id": MODEL_ID,
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "transformers_cache_only": True,
        "prompt": PROMPT,
        "generated_text": generated_text,
    }

    output_path = Path("logs/base_model_check.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
