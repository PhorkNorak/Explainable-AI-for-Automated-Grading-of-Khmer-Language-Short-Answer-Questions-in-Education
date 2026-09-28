"""Merge the three tested Pintu QAR LoRA adapters into complete BF16 models.

Run from the repository root on the HPC machine. Each model is loaded, merged,
saved, and released from GPU memory before the next model is processed.

Two merge targets are supported (benchmarked against each other in exp15):

* ``bf16`` (M1, default): adapter + the original full-precision base ``W``.
  Standard community practice (merge in 16-bit, quantize last).
* ``nf4-dequant`` (M2): adapter + ``dequant(NF4(W))`` stored in BF16. QLoRA
  trained the adapter against the 4-bit base, so this reproduces the
  training-time function most closely (the approach Unsloth's
  ``save_pretrained_merged(..., "merged_16bit")`` takes).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from safetensors import safe_open
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _release_manifest import StageManifest  # noqa: E402

MERGE_TARGETS = ("bf16", "nf4-dequant")
DEFAULT_OUTPUT_ROOTS = {
    "bf16": Path("publish/full_models"),
    "nf4-dequant": Path("publish/full_models_nf4dequant"),
}


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


def exact_adapter_targets(adapter_path: Path, base_model: torch.nn.Module) -> list[str]:
    """Map saved LoRA tensors to exact module paths in the complete base model.

    Gemma 4 contains text, audio, and vision modules that reuse generic names
    such as ``q_proj``.  Loading the adapter's original broad target list would
    therefore make PEFT try to wrap unrelated ``Gemma4ClippableLinear`` audio
    and vision layers.  The adapter checkpoint itself identifies precisely
    which language-model modules were trained, so use those paths instead.
    """
    weights_path = adapter_path / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing adapter weights: {weights_path}")

    with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())

    saved_modules: set[str] = set()
    for key in keys:
        for marker in (".lora_A.", ".lora_B."):
            if marker in key:
                saved_modules.add(key.split(marker, 1)[0])
                break
    if not saved_modules:
        raise RuntimeError(f"No LoRA tensors found in {weights_path}")

    base_modules = tuple(name for name, _ in base_model.named_modules() if name)
    exact_targets: set[str] = set()
    unmatched: list[str] = []
    for saved_name in sorted(saved_modules):
        matches = [
            name
            for name in base_modules
            if saved_name == name or saved_name.endswith("." + name)
        ]
        if not matches:
            unmatched.append(saved_name)
            continue
        exact_targets.add(max(matches, key=len))

    if unmatched:
        preview = "\n  ".join(unmatched[:10])
        raise RuntimeError(
            "Adapter modules do not match the selected complete base model:\n  "
            + preview
        )
    if len(exact_targets) != len(saved_modules):
        raise RuntimeError(
            "Adapter-to-base module mapping was not one-to-one: "
            f"{len(saved_modules)} saved modules, {len(exact_targets)} targets"
        )

    return sorted(exact_targets)


def training_base_id(spec: ModelSpec) -> str:
    """The base the adapter was actually trained on (from adapter_config.json).

    Unsloth may redirect a plain HF id to a pre-quantized ``unsloth/...bnb-4bit``
    derivative; that repo carries the exact NF4 quantization used in training.
    """
    config = json.loads((spec.adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    return config.get("base_model_name_or_path") or spec.base_id


def load_nf4_dequantized(spec: ModelSpec):
    """Load the base in NF4 exactly as in training, then dequantize to BF16.

    A pre-quantized training base (its config has a quantization_config) is
    loaded as stored. Otherwise the upstream base is quantized with the same
    BitsAndBytesConfig as ``exp08_llm_finetune.try_load_hf`` (NF4, double
    quantization, BF16 compute), which is also Unsloth's default 4-bit recipe.
    ``dequantize()`` then replaces every Linear4bit by a BF16 Linear holding
    dequant(NF4(W)).
    """
    source = training_base_id(spec)
    kwargs = dict(device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
    model, prequantized = None, False
    if source != spec.base_id:
        try:
            model = AutoModelForMultimodalLM.from_pretrained(source, dtype=torch.bfloat16, **kwargs)
            prequantized = bool(getattr(model, "is_quantized", False))
        except Exception as error:  # e.g. a pre-quantized repo this class cannot load
            print(f"Could not load training base {source}: {error}")
    if not prequantized:
        if model is not None:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        source = f"{spec.base_id} (NF4, double quant, bf16 compute)"
        model = AutoModelForMultimodalLM.from_pretrained(
            spec.base_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            dtype=torch.bfloat16,
            **kwargs,
        )
    model = model.dequantize()
    model = model.to(torch.bfloat16)
    if getattr(model.config, "quantization_config", None) is not None:
        del model.config.quantization_config
    remaining = [n for n, m in model.named_modules() if "4bit" in type(m).__name__]
    if remaining:
        raise RuntimeError(f"Modules still quantized after dequantize(): {remaining[:5]}")
    print(f"Dequantized NF4 base to BF16 from: {source}", flush=True)
    return model, source


def merge_one(spec: ModelSpec, output_root: Path, merge_target: str,
              manifest: StageManifest) -> Path:
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
    print(f"Target:  {merge_target}")
    print(f"Output:  {output_path}")
    print("=" * 72)

    processor = AutoProcessor.from_pretrained(
        spec.base_id,
        trust_remote_code=True,
    )
    if merge_target == "bf16":
        base_model = AutoModelForMultimodalLM.from_pretrained(
            spec.base_id,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        manifest.extra["merge_base"] = spec.base_id
    else:
        base_model, quant_source = load_nf4_dequantized(spec)
        manifest.extra["merge_base"] = quant_source
    peft_config = PeftConfig.from_pretrained(str(spec.adapter_path))
    peft_config.target_modules = exact_adapter_targets(spec.adapter_path, base_model)
    print(f"Exact trained LoRA targets: {len(peft_config.target_modules)}", flush=True)
    manifest.metrics["lora_target_modules"] = len(peft_config.target_modules)
    manifest.extra.update({
        "merge_target": merge_target,
        "training_base": training_base_id(spec),
        "lora_r": getattr(peft_config, "r", None),
        "lora_alpha": getattr(peft_config, "lora_alpha", None),
    })
    manifest.add_inputs(spec.adapter_path)
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(spec.adapter_path),
        is_trainable=False,
        config=peft_config,
    )
    peft_model.eval()

    merged_model = peft_model.merge_and_unload(
        safe_merge=True,
        progressbar=True,
    )
    merged_model.eval()

    # The HPC login node has limited host RAM.  Drop the now-empty PEFT wrapper
    # before serialization and use small shards so copying GPU tensors to CPU
    # never requires a multi-gigabyte temporary buffer.
    del peft_model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    output_path.mkdir(parents=True)
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="1GB",
    )
    processor.save_pretrained(output_path)

    weight_files = list(output_path.glob("model*.safetensors"))
    adapter_files = list(output_path.glob("adapter_*"))
    if not weight_files:
        raise RuntimeError(f"No complete model weights were saved in {output_path}")
    if adapter_files:
        raise RuntimeError(f"Unexpected adapter-only files in {output_path}: {adapter_files}")

    print(f"Saved complete BF16 model: {output_path}")
    manifest.add_outputs(output_path)
    manifest.metrics["weight_gb"] = round(
        sum(p.stat().st_size for p in weight_files) / 1e9, 3
    )

    del merged_model
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
        "--merge-target",
        choices=MERGE_TARGETS,
        default="bf16",
        help="bf16: original base weights (M1). nf4-dequant: dequantized NF4 base (M2).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for complete merged-model folders "
             "(default: publish/full_models or publish/full_models_nf4dequant).",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip SHA-256 of the saved weights in the manifest (faster).",
    )
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = DEFAULT_OUTPUT_ROOTS[args.merge_target]
    return args


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    variant = "merged_bf16" if args.merge_target == "bf16" else "merged_nf4dequant"
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        with StageManifest("merge", spec.release_name, variant, args=vars(args),
                           hash_files=not args.no_hash) as manifest:
            merge_one(spec, args.output_root, args.merge_target, manifest)
    print("\nAll requested Pintu models were merged successfully.")


if __name__ == "__main__":
    main()
