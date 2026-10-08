"""R3: 学習用の環境の確認。版、GPU、高速化の部品（fla、causal-conv1d）が実際に GPU で動くか、Qwen3.5 が読み込めるか。"""
import importlib
import time

import torch
import transformers

print("torch", torch.__version__, "| cuda", torch.version.cuda, "| transformers", transformers.__version__)
print("gpu:", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0), "| memory GB", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
for m in ("fla", "causal_conv1d", "flash_attn", "accelerate", "huggingface_hub", "PIL"):
    try:
        mod = importlib.import_module(m)
        print(" ", m, getattr(mod, "__version__", "ok"))
    except Exception as e:  # noqa: BLE001
        print(" ", m, "MISSING", type(e).__name__)

# causal-conv1d が GPU で動くか
try:
    from causal_conv1d import causal_conv1d_fn

    x = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16)
    y = causal_conv1d_fn(x, w, None, activation="silu")
    torch.cuda.synchronize()
    print("causal_conv1d kernel on GPU: OK", tuple(y.shape))
except Exception as e:  # noqa: BLE001
    print("causal_conv1d kernel on GPU: FAILED", type(e).__name__, str(e)[:200])

# fla の Gated DeltaNet の部品が GPU で動くか
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    B, T, H, D = 1, 256, 4, 64
    q = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, D, device="cuda", dtype=torch.float32), dim=-1).to(torch.bfloat16)
    v = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.float32))
    beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16)
    t0 = time.time()
    o, _ = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=False)
    torch.cuda.synchronize()
    print("fla chunk_gated_delta_rule on GPU: OK", tuple(o.shape), f"{time.time() - t0:.1f}s（初回はコンパイルを含む）")
except Exception as e:  # noqa: BLE001
    print("fla chunk_gated_delta_rule on GPU: FAILED", type(e).__name__, str(e)[:300])

from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401

print("Qwen3_5ForConditionalGeneration import: OK")
try:
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq

    print("modeling fast path flags:", {k: getattr(mq, k) for k in dir(mq) if "fast_path" in k.lower() or k.startswith("is_flash_linear") or k in ("causal_conv1d_fn", "chunk_gated_delta_rule") and False})
except Exception as e:  # noqa: BLE001
    print("modeling inspect failed", e)
