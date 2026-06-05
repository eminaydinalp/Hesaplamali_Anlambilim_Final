from __future__ import annotations

import json
import platform
from pathlib import Path

import torch


def get_device_info() -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    mps_available = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )

    if cuda_available:
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
    elif mps_available:
        device = "mps"
        device_name = "Apple Metal Performance Shaders"
        device_count = 1
    else:
        device = "cpu"
        device_name = platform.processor() or platform.machine()
        device_count = 1

    return {
        "device": device,
        "device_name": device_name,
        "device_count": device_count,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def main() -> None:
    info = get_device_info()
    output_path = Path("logs/device_check.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
