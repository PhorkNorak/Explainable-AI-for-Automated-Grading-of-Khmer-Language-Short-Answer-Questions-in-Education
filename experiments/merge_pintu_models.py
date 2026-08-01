"""Merge the three tested Pintu QAR LoRA adapters into complete BF16 models.

Run from the repository root on the HPC machine. Each model is loaded, merged,
saved, and released from GPU memory before the next model is processed.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor


@dataclass(frozen=True)
class ModelSpec:
    base_id: str
    adapter_path: Path
    release_name: str


MODEL_SPECS = {
    "qwen": ModelSpec(
        base_id="Qwen/Qwen3.5-4B",
        adapter_path=Path(
            "results_no10c_v08_llm_qwen35_4b/"
            "runs/clean_qar_qwen35_4b/lora_adapter"
        ),
        release_name="Pintu-Qwen3.5-4B",
    ),
    "gemma": ModelSpec(
        # Merge the adapter trained on Unsloth's 4-bit derivative into its
        # non-quantized upstream instruction model.
        base_id="google/gemma-4-E4B-it",
        adapter_path=Path(
            "results_no10c_v08_llm_gemma4_e4b/"
            "runs/clean_qar_gemma4_e4b/lora_adapter"
        ),
        release_name="Pintu-Gemma4-E4B",
    ),
    "sealion": ModelSpec(
        base_id="aisingapore/Gemma-SEA-LION-v4.5-E2B-IT",
        adapter_path=Path(
            "results_no10c_v08_llm_sealion_v45_e2b/"
            "runs/clean_qar_sealion_v45_e2b/lora_adapter"
        ),
        release_name="Pintu-SEA-LION-v4.5-E2B",
    ),
}


def merge_one(spec: ModelSpec, output_root: Path) -> Path:
    adapter_config = spec.adapter_path / "adapter_config.json"
    if not adapter_config.is_file():
        raise FileNotFoundError(f"Missing adapter configuration: {adapter_config}")

    output_path = output_root / spec.release_name
    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}. Move or inspect it before rerunning."
        )

    print("\n" + "=" * 72)
    print(f"Release: {spec.release_name}")
    print(f"Base:    {spec.base_id}")
    print(f"Adapter: {spec.adapter_path}")
    print(f"Output:  {output_path}")
    print("=" * 72)

    processor = AutoProcessor.from_pretrained(
        spec.base_id,
        trust_remote_code=True,
    )
    base_model = AutoModelForMultimodalLM.from_pretrained(
        spec.base_id,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(spec.adapter_path),
        is_trainable=False,
    )
    peft_model.eval()

    merged_model = peft_model.merge_and_unload(
        safe_merge=True,
        progressbar=True,
    )
    merged_model.eval()

    output_path.mkdir(parents=True)
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    processor.save_pretrained(output_path)

    weight_files = list(output_path.glob("model*.safetensors"))
    adapter_files = list(output_path.glob("adapter_*"))
    if not weight_files:
        raise RuntimeError(f"No complete model weights were saved in {output_path}")
    if adapter_files:
        raise RuntimeError(f"Unexpected adapter-only files in {output_path}: {adapter_files}")

    print(f"Saved complete BF16 model: {output_path}")

    del merged_model
    del peft_model
    del base_model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(MODEL_SPECS),
        help="Models to merge sequentially (default: all three).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("publish/full_models"),
        help="Directory for complete merged-model folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for model_key in args.models:
        merge_one(MODEL_SPECS[model_key], args.output_root)
    print("\nAll requested Pintu models were merged successfully.")


if __name__ == "__main__":
    main()
