"""Validate complete Pintu models on the report's curated QAR test split.

Run one model at a time after ``merge_pintu_models.py``.  The script reproduces
the original prompt, preprocessing, split, generation, and metrics, then
compares the merged model's raw-score predictions with the saved QLoRA-adapter
run used for the report.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
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


def predict(model, processor, test_df: pd.DataFrame) -> pd.DataFrame:
    device = next(model.parameters()).device
    scores_normalized: list[float] = []
    scores_raw: list[int] = []
    raw_outputs: list[str] = []

    for row in tqdm(test_df.to_dict("records"), desc="Scoring test answers"):
        prompt = exp08.render_prompt_text(processor, row, with_answer=False)
        score, raw_output = exp08.generate_score(
            model,
            processor,
            prompt,
            int(row["Max Score"]),
            device,
        )
        scores_raw.append(score)
        scores_normalized.append(score / max(int(row["Max Score"]), 1))
        raw_outputs.append(raw_output)

    predictions = test_df.copy()
    predictions["pred_score"] = scores_normalized
    predictions["pred_raw"] = scores_raw
    predictions["llm_raw_output"] = raw_outputs
    return predictions


def validate_one(model_key: str, model_root: Path, output_root: Path) -> None:
    spec = SPECS[model_key]
    model_path = model_root / spec.release_name
    if not model_path.is_dir():
        raise FileNotFoundError(f"Missing complete model: {model_path}")

    print("\n" + "=" * 72, flush=True)
    print(f"Validating: {spec.release_name}", flush=True)
    print(f"Model:      {model_path}", flush=True)
    print(f"Reference:  {spec.original_run}", flush=True)
    print("=" * 72, flush=True)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else "cpu"
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

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
    test_df = data.apply_preprocess(test_df, "clean")
    print(f"Curated test answers: {len(test_df)}", flush=True)

    predictions = predict(model, processor, test_df)
    metrics = exp08.llm_metrics(predictions)

    output_dir = output_root / spec.release_name
    output_dir.mkdir(parents=True, exist_ok=True)
    exp08.write_predictions(predictions, str(output_dir), "test")

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

    report = {
        "model": spec.release_name,
        "model_path": str(model_path),
        "input": "qar",
        "preprocess": "clean",
        "test_rows_source": str(test_rows_path),
        "test_rows_source_predictions_not_used": True,
        "test_answers": len(test_df),
        "merged_test_metrics": metrics,
        "adapter_comparison": comparison,
    }
    with (output_dir / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=float)

    print("\nMerged-model test metrics:", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=float), flush=True)
    print("\nAdapter comparison:", flush=True)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    print(f"\nValidation saved to: {output_dir / 'validation.json'}", flush=True)

    del predictions
    del test_df
    del test_rows
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(SPECS),
        default=["qwen"],
        help="Complete models to validate sequentially (default: qwen).",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("publish/full_models"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("publish/validation"),
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    for model_key in args.models:
        validate_one(model_key, args.model_root, args.output_root)
    print("\nAll requested Pintu validations completed.", flush=True)


if __name__ == "__main__":
    main()
