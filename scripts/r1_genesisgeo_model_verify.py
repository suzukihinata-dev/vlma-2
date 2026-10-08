"""R1 smoke check of the published GenesisGeo-2B checkpoint on WNPC GPU."""

import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from newclid.evaluation.search_runtime import try_full_aux_dsl_to_constructions
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MODEL_REPO = "ZJUVAI/GenesisGeo-2B"
MODEL_REVISION = "3295483676aa256cbc90b013adae97a1bb45e043"
MODEL_DIR = Path("/models/genesisgeo-2b")
ARTIFACT_DIR = Path("/artifacts/genesisgeo")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = ARTIFACT_DIR / "model-result.txt"
    result_path.write_text("Model check running; previous success does not apply.\n")
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=True,
        capture_output=True,
        text=True,
    )
    (ARTIFACT_DIR / "model-pip-check.log").write_text(pip_check.stdout)
    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    (ARTIFACT_DIR / "model-pip-freeze.txt").write_text(pip_freeze.stdout)

    assert torch.cuda.is_available(), "CUDA device is not available"
    gpu_name = torch.cuda.get_device_name(0)
    local_path = Path(
        snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            local_dir=str(MODEL_DIR),
        )
    )
    weights = local_path / "model.safetensors"
    assert weights.is_file() and weights.stat().st_size > 0
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    processor = AutoProcessor.from_pretrained(str(local_path))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(local_path),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()

    generated_paths = glob.glob("/artifacts/genesisgeo/generated/**/*.jsonl", recursive=True)
    assert generated_paths, "R1 generated problems are missing"
    newest_path = max(generated_paths, key=os.path.getmtime)
    record = json.loads(Path(newest_path).read_text().splitlines()[0])
    image_path = Path(record["image_path"])
    assert image_path.is_file(), image_path
    with Image.open(image_path) as image:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": record["llm_input_renamed"]},
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    response = processor.decode(
        output_ids[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()
    assert response, "Model returned an empty response"
    aux_match = re.search(r"<aux>(.*?)</aux>", response, re.DOTALL)
    assert aux_match is not None, "Model did not return an auxiliary-construction block"
    parsed_aux = try_full_aux_dsl_to_constructions(aux_match.group(1).strip())
    assert parsed_aux is not None, "GenesisGeo could not parse the model's auxiliary construction"

    report = {
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "weights_sha256": digest.hexdigest(),
        "weights_bytes": weights.stat().st_size,
        "gpu": gpu_name,
        "torch": torch.__version__,
        "input": record["llm_input_renamed"],
        "image_path": str(image_path),
        "response": response,
        "parsed_aux_construction": parsed_aux,
    }
    (ARTIFACT_DIR / "model-smoke.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    result_path.write_text(
        "GenesisGeo-2B loaded on GPU and produced a parseable auxiliary construction from vision+text.\n"
    )
    print(result_path.read_text().strip())
    print(f"parsed_aux_construction={parsed_aux}")


if __name__ == "__main__":
    main()
