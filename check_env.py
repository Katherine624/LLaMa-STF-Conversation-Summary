from __future__ import annotations

import platform

import accelerate
import bitsandbytes
from bitsandbytes import cextension as bnb_cextension
import datasets
import peft
import torch
import transformers


def main() -> None:
    print(f"Python:       {platform.python_version()}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"PEFT:         {peft.__version__}")
    print(f"Accelerate:   {accelerate.__version__}")
    print(f"Datasets:     {datasets.__version__}")
    print(f"bitsandbytes: {bitsandbytes.__version__}")
    bnb_lib = getattr(bnb_cextension, "lib", None)
    if bnb_lib is None:
        raise SystemExit("bitsandbytes native CUDA library did not load.")
    print(f"bnb native:   {type(bnb_lib).__name__}")
    print(f"CUDA usable:  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Check the NVIDIA driver and PyTorch build.")

    props = torch.cuda.get_device_properties(0)
    print(f"GPU:          {props.name}")
    print(f"VRAM:         {props.total_memory / 1024**3:.2f} GiB")
    print(f"Capability:   sm_{props.major}{props.minor}")

    x = torch.tensor([1.0, 2.0], device="cuda")
    print(f"CUDA test:    {(x * 2).tolist()}")
    print("Environment check passed.")


if __name__ == "__main__":
    main()

