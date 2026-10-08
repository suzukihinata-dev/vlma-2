import importlib.util
import sys

import torch
import transformers

print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers has qwen3_5:", importlib.util.find_spec("transformers.models.qwen3_5") is not None)
for m in ("fla", "causal_conv1d", "flash_attn", "peft", "trl", "accelerate", "datasets", "deepspeed"):
    print(f"  {m}:", importlib.util.find_spec(m) is not None)
