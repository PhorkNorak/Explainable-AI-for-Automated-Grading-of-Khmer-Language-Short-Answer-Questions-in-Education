"""Validate complete Pintu models on the report's curated QAR test split.

Run one model at a time after ``merge_pintu_models.py``.  The script reproduces
the original prompt, preprocessing, split, generation, and metrics, then
compares the merged model's raw-score predictions with the saved QLoRA-adapter
run used for the report.

Outputs follow the shared release-benchmark layout read by
``exp15_pintu_release_benchmark.py``::

    publish/benchmark/<release>/<variant>__<device>/
        predictions_test.csv   validation.json

with variant ``adapter_qlora4bit``, ``merged_bf16`` (M1) or
``merged_nf4dequant`` (M2), plus a manifest in ``publish/manifests/``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (str(ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import data  # noqa: E402
import exp08_llm_finetune as exp08  # noqa: E402
from _release_manifest import StageManifest  # noqa: E402


@dataclass(frozen=True)
class ValidationSpec:
    release_name: str
    original_run: Path
    test_rows_run: Path


SPECS = {
    "qwen": ValidationSpec(
        release_name="Pintu-Qwen3.5-4B",
        original_run=Path(
            "results_no10c_v08_llm_qwen35_4b/"
            "runs/clean_qar_qwen35_4b"
        ),
        test_rows_run=Path(
            "results_no10c_v08_llm_qwen35_4b/"
            "runs/clean_ra_qwen35_4b"
        ),
    ),
    "gemma": ValidationSpec(
        release_name="Pintu-Gemma4-E4B",
        original_run=Path(
            "results_no10c_v08_llm_gemma4_e4b/"
            "runs/clean_qar_gemma4_e4b"
        ),
        test_rows_run=Path(
            "results_no10c_v08_llm_gemma4_e4b/"
            "runs/clean_ra_gemma4_e4b"
        ),
    ),
    "sealion": ValidationSpec(
        release_name="Pintu-SEA-LION-v4.5-E2B",
        original_run=Path(
            "results_no10c_v08_llm_sealion_v45_e2b/"
            "runs/clean_qar_sealion_v45_e2b"
        ),
        test_rows_run=Path(
            "results_no10c_v08_llm_sealion_v45_e2b/"
            "runs/clean_ra_sealion_v45_e2b"
        ),
    ),
}

MERGED_ROOTS = {
    "bf16": Path("publish/full_models"),
    "nf4-dequant": Path("publish/full_models_nf4dequant"),
}
MERGED_VARIANTS = {"bf16": "merged_bf16", "nf4-dequant": "merged_nf4dequant"}


def load_test_rows(spec: ValidationSpec) -> tuple[pd.DataFrame, Path]:
    """The report's curated test rows, preprocessed exactly as in exp08."""
    test_rows_path = spec.test_rows_run / "predictions_test.csv"
    if not test_rows_path.is_file():
        raise FileNotFoundError(
            f"Missing saved test rows: {test_rows_path}"
        )
    test_rows = pd.read_csv(test_rows_path)
    required_columns = [
        "Question",
        "Reference",
        "Answer",
        "Max Score",
        "true_raw",
        "true_label",
        "true_score",
    ]
    missing_columns = [column for column in required_columns if column not in test_rows]
    if missing_columns:
        raise ValueError(
            f"Missing columns in {test_rows_path}: {missing_columns}"
        )
    test_df = test_rows[required_columns].rename(
        columns={
            "true_raw": "Student Score",
            "true_label": "score_label",
            "true_score": "normalized_score",
        }
    )
    return data.apply_preprocess(test_df, "clean"), test_rows_path


def compare_with_original(spec: ValidationSpec, predictions: pd.DataFrame,
                          metrics: dict) -> dict:
    """Agreement of these predictions with the saved QLoRA-adapter report run."""
    comparison: dict[str, object] = {}
    original_predictions_path = spec.original_run / "predictions_test.csv"
    if not original_predictions_path.is_file():
        comparison["prediction_comparison_available"] = False
        comparison["missing_original_qar_predictions"] = str(
            original_predictions_path
        )
    else:
        original = pd.read_csv(original_predictions_path)
        if len(original) != len(predictions):
            comparison["prediction_comparison_available"] = False
            comparison["original_prediction_rows"] = len(original)
            comparison["merged_prediction_rows"] = len(predictions)
        else:
            original_raw = original["pred_raw"].to_numpy(dtype=int)
            merged_raw = predictions["pred_raw"].to_numpy(dtype=int)
            comparison.update(
                {
                    "prediction_comparison_available": True,
                    "exact_prediction_matches": int(np.sum(original_raw == merged_raw)),
                    "prediction_rows": len(merged_raw),
                    "prediction_match_rate": float(np.mean(original_raw == merged_raw)),
                    "mean_absolute_raw_difference": float(
                        np.mean(np.abs(original_raw - merged_raw))
                    ),
                    "maximum_absolute_raw_difference": int(
                        np.max(np.abs(original_raw - merged_raw))
                    ),
                }
            )

    original_metrics_path = spec.original_run / "metrics.json"
    if original_metrics_path.is_file():
        with original_metrics_path.open(encoding="utf-8") as handle:
            original_metrics = json.load(handle).get("test", {})
        comparison["original_test_qwk"] = original_metrics.get("qwk")
        if original_metrics.get("qwk") is not None:
            comparison["qwk_difference"] = float(
                metrics["qwk"] - float(original_metrics["qwk"])
            )
    return comparison


def latency_summary(seconds: list[float]) -> dict:
    values = np.asarray(seconds, dtype=float)
    return {
        "seconds_per_answer_median": float(np.median(values)),
        "seconds_per_answer_p90": float(np.percentile(values, 90)),
        "seconds_per_answer_mean": float(values.mean()),
        "seconds_total": float(values.sum()),
    }


def predict(model, processor, test_df: pd.DataFrame) -> tuple[pd.DataFrame, list[float], int]:
    device = next(model.parameters()).device
    scores_normalized: list[float] = []
    scores_raw: list[int] = []
    raw_outputs: list[str] = []
    seconds: list[float] = []
    parse_failures = 0

    for row in tqdm(test_df.to_dict("records"), desc="Scoring test answers"):
        prompt = exp08.render_prompt_text(processor, row, with_answer=False)
        start = time.perf_counter()
        score, raw_output = exp08.generate_score(
            model,
            processor,
            prompt,
            int(row["Max Score"]),
            device,
        )
        seconds.append(time.perf_counter() - start)
        parse_failures += int(not exp08.parse_score(raw_output, int(row["Max Score"]))[1])
        scores_raw.append(score)
        scores_normalized.append(score / max(int(row["Max Score"]), 1))
        raw_outputs.append(raw_output)

    predictions = test_df.copy()
    predictions["pred_score"] = scores_normalized
    predictions["pred_raw"] = scores_raw
    predictions["llm_raw_output"] = raw_outputs
    return predictions, seconds, parse_failures


def validate_one(
    model_key: str,
    model_root: Path,
    output_root: Path,
    processor_source: str,
    model_source: str,
    merge_target: str,
    manifest: StageManifest,
) -> None:
    spec = SPECS[model_key]
    adapter_path = spec.original_run / "lora_adapter"
    model_path = (
        adapter_path
        if model_source == "adapter"
        else model_root / spec.release_name
    )
    if not model_path.is_dir():
        raise FileNotFoundError(f"Missing model source: {model_path}")

    print("\n" + "=" * 72, flush=True)
    print(f"Validating: {spec.release_name}", flush=True)
    print(f"Model:      {model_path}", flush=True)
    print(f"Reference:  {spec.original_run}", flush=True)
    print("=" * 72, flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    if model_source == "adapter":
        from unsloth import FastLanguageModel

        processor_path = adapter_path
        print(f"Processor:  {processor_path}", flush=True)
        model, processor = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    else:
        processor_path = (
            adapter_path if processor_source == "adapter" else model_path
        )
        if not processor_path.is_dir():
            raise FileNotFoundError(f"Missing processor source: {processor_path}")
        print(f"Processor:  {processor_path}", flush=True)
        processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
        )
        device_map = (
            {"": torch.cuda.current_device()} if torch.cuda.is_available() else "cpu"
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    model.train(False)
    load_seconds = time.perf_counter() - load_start

    test_df, test_rows_path = load_test_rows(spec)
    print(f"Curated test answers: {len(test_df)}", flush=True)

    predictions, seconds, parse_failures = predict(model, processor, test_df)
    metrics = exp08.llm_metrics(predictions)

    device = "gpu" if torch.cuda.is_available() else "cpu"
    variant = "adapter_qlora4bit" if model_source == "adapter" else MERGED_VARIANTS[merge_target]
    output_dir = output_root / spec.release_name / f"{variant}__{device}"
    output_dir.mkdir(parents=True, exist_ok=True)
    exp08.write_predictions(predictions, str(output_dir), "test")

    comparison = compare_with_original(spec, predictions, metrics)
    cost = {
        "device": device,
        "load_seconds": float(load_seconds),
        "parse_failures": int(parse_failures),
        **latency_summary(seconds),
    }
    if torch.cuda.is_available():
        cost["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        cost["gpu"] = torch.cuda.get_device_name(0)
    weight_files = list(model_path.glob("*.safetensors"))
    cost["weights_gb"] = round(sum(p.stat().st_size for p in weight_files) / 1e9, 3)

    report = {
        "model": spec.release_name,
        "variant": variant,
        "model_source": model_source,
        "merge_target": None if model_source == "adapter" else merge_target,
        "model_path": str(model_path),
        "processor_path": str(processor_path),
        "runtime": "unsloth" if model_source == "adapter" else "transformers",
        "input": "qar",
        "preprocess": "clean",
        "decoding": {"greedy": True, "max_new_tokens": 32},
        "test_rows_source": str(test_rows_path),
        "test_rows_source_predictions_not_used": True,
        "test_answers": len(test_df),
        "merged_test_metrics": metrics,
        "test_metrics": metrics,
        "adapter_comparison": comparison,
        "cost": cost,
    }
    with (output_dir / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=float)

    manifest.variant = f"{variant}__{device}"
    manifest.path = manifest.path.with_name(
        f"validate_{spec.release_name}_{variant}__{device}.json"
    )
    manifest.add_inputs(test_rows_path)
    manifest.add_outputs(output_dir)
    manifest.metrics.update({**metrics, **cost})

    print("\nTest metrics:", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=float), flush=True)
    print("\nAdapter comparison:", flush=True)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    print(f"\nValidation saved to: {output_dir / 'validation.json'}", flush=True)

    del predictions
    del test_df
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(SPECS),
        default=["qwen"],
        help="Complete models to validate sequentially (default: qwen).",
    )
    parser.add_argument(
        "--merge-target",
        choices=tuple(MERGED_ROOTS),
        default="bf16",
        help="Which merged build to validate (M1 bf16 or M2 nf4-dequant).",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="Folder holding merged models (default follows --merge-target).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("publish/benchmark"),
    )
    parser.add_argument(
        "--processor-source",
        choices=("model", "adapter"),
        default="model",
        help="Load processor files from the merged model or original adapter.",
    )
    parser.add_argument(
        "--model-source",
        choices=("merged", "adapter"),
        default="merged",
        help="Validate complete merged weights or the original 4-bit QLoRA adapter.",
    )
    args = parser.parse_args()
    if args.model_root is None:
        args.model_root = MERGED_ROOTS[args.merge_target]
    return args


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(0)
    args = parse_args()
    for model_key in args.models:
        with StageManifest("validate", SPECS[model_key].release_name,
                           args=vars(args)) as manifest:
            validate_one(
                model_key,
                args.model_root,
                args.output_root,
                args.processor_source,
                args.model_source,
                args.merge_target,
                manifest,
            )
    print("\nAll requested Pintu validations completed.", flush=True)


if __name__ == "__main__":
    main()
