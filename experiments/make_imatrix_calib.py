"""Build an importance-matrix (imatrix) calibration text for GGUF quantization.

llama.cpp's ``llama-imatrix`` measures which weights matter most on a
calibration text; low-bit quants (Q4_K_M, Q5_K_M) then keep those weights more
precise. To avoid test leakage the calibration uses the **training split only**
(the same no10c split exp08 trained on), rendered with the exact chat-template
grading prompt plus the gold score, i.e. the text the adapter was trained on.
Any training row whose (question, answer) text also occurs in the saved test
rows is dropped from the calibration (identical short answers from different
students can collide), and the number dropped is logged in the manifest.

    python experiments/make_imatrix_calib.py --models qwen
    -> publish/gguf/Pintu-Qwen3.5-4B/imatrix_calibration.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (str(ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from _release_manifest import StageManifest  # noqa: E402

EXP_SUFFIX = {
    "qwen": "v08_llm_qwen35_4b",
    "gemma": "v08_llm_gemma4_e4b",
    "sealion": "v08_llm_sealion_v45_e2b",
}


def training_rows(model_key: str) -> pd.DataFrame:
    """The no10c training split exactly as exp08 builds it."""
    import importlib

    import data
    from _common import DATASETS, patch_config

    dataset = next(d for d in DATASETS if d["run_name"] == "no10c")
    patch_config(dataset["run_name"], dataset["drop_zero"],
                 exp_suffix=EXP_SUFFIX[model_key], raw_csv=dataset["raw_csv"])
    importlib.reload(data)
    train_df, _, _ = data.split_dataframe(data.load_dataframe())
    return data.apply_preprocess(train_df, "clean")


def drop_test_text(train_df: pd.DataFrame, test_rows_path: Path) -> tuple[pd.DataFrame, int, int]:
    """Remove training rows whose (Question, Answer) text occurs in the test rows."""
    test = pd.read_csv(test_rows_path)
    key = ["Question", "Answer"]
    test_keys = set(map(tuple, test[key].astype(str).to_numpy()))
    collide = train_df[key].astype(str).apply(tuple, axis=1).isin(test_keys)
    return train_df[~collide.to_numpy()], int(collide.sum()), len(test)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=tuple(EXP_SUFFIX), default=["qwen"])
    parser.add_argument("--processor-root", type=Path, default=Path("publish/full_models"))
    parser.add_argument("--output-root", type=Path, default=Path("publish/gguf"))
    args = parser.parse_args()

    from transformers import AutoProcessor

    import exp08_llm_finetune as exp08
    from validate_pintu_models import SPECS

    exp08._set_input_fmt("qar")
    for model_key in args.models:
        spec = SPECS[model_key]
        with StageManifest("imatrix_calib", spec.release_name, args=vars(args)) as manifest:
            train_df = training_rows(model_key)
            test_path = spec.test_rows_run / "predictions_test.csv"
            if not test_path.is_file():  # same split, saved by the qar run
                test_path = spec.original_run / "predictions_test.csv"
            train_df, n_dropped, n_test = drop_test_text(train_df, test_path)
            processor = AutoProcessor.from_pretrained(args.processor_root / spec.release_name,
                                                      trust_remote_code=True)
            texts = [exp08.render_prompt_text(processor, row, with_answer=True)
                     for row in train_df.to_dict("records")]
            out_dir = args.output_root / spec.release_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "imatrix_calibration.txt"
            out_path.write_text("\n\n".join(texts), encoding="utf-8")
            manifest.add_outputs(out_path)
            manifest.metrics.update({"train_rows": len(train_df), "test_rows_checked": n_test,
                                     "dropped_text_collisions": n_dropped})
            print(f"{spec.release_name}: {len(texts)} training prompts "
                  f"({n_dropped} dropped as test-text collisions) -> {out_path}")


if __name__ == "__main__":
    main()
