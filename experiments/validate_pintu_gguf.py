"""Validate text-only GGUF builds of a Pintu model on the report's test split.

Mirrors ``validate_pintu_models.py`` but runs the quantized GGUF files with
llama.cpp (``llama-cpp-python``), so the same curated test rows, preprocessing,
chat-template prompt (thinking disabled), greedy decoding, score parser and
metrics are applied to every published variant. Run on the HPC (GPU) and/or a
CPU machine to fill both latency axes::

    python experiments/validate_pintu_gguf.py --models qwen --device gpu
    python experiments/validate_pintu_gguf.py --models qwen --device cpu --threads 8

Each GGUF file ``<release>-<QTYPE>[-imat].gguf`` in ``publish/gguf/<release>/``
becomes variant ``gguf_<qtype>[_imat]`` and writes
``publish/benchmark/<release>/<variant>__<device>/{predictions_test.csv,validation.json}``.

``--check-text-only FILE`` only inspects a GGUF and fails if it contains any
vision/audio tensors (used by ``quantize_pintu_gguf.sh``).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (str(ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from _release_manifest import StageManifest, peak_rss_gb  # noqa: E402

# llama.cpp tensor-name prefixes of non-text towers: vision encoder ("v."),
# audio encoder ("a."), and multimodal projectors ("mm.").
NON_TEXT_PREFIXES = ("v.", "a.", "mm.")
QTYPE_TAIL = re.compile(r"^(?P<qtype>[A-Za-z0-9_]+)(?P<imat>-imat)?\.gguf$")


def non_text_tensors(tensor_names) -> list[str]:
    return [name for name in tensor_names if name.startswith(NON_TEXT_PREFIXES)]


def check_text_only(path: Path) -> dict:
    """Read GGUF metadata; raise if any vision/audio tensor is present."""
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    names = [tensor.name for tensor in reader.tensors]
    bad = non_text_tensors(names)
    arch_field = reader.fields.get("general.architecture")
    arch = None
    if arch_field is not None:
        arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8", "replace")
    summary = {"file": str(path), "architecture": arch, "tensors": len(names),
               "non_text_tensors": len(bad)}
    print(json.dumps(summary, indent=2))
    if bad:
        raise SystemExit(f"{path} is not text-only; first non-text tensors: {bad[:5]}")
    return summary


def parse_gguf_name(path: Path, release: str) -> tuple[str, bool]:
    """``<release>-<QTYPE>[-imat].gguf`` -> (QTYPE, used_imatrix)."""
    prefix = f"{release}-"
    match = QTYPE_TAIL.match(path.name[len(prefix):]) if path.name.startswith(prefix) else None
    if not match:
        raise ValueError(f"Unexpected GGUF file name for {release}: {path.name}")
    return match["qtype"].upper(), bool(match["imat"])


def variant_from_filename(path: Path, release: str) -> str:
    qtype, imat = parse_gguf_name(path, release)
    return f"gguf_{qtype.lower()}" + ("_imat" if imat else "")


class GGUFScorer:
    """Greedy integer scoring with llama.cpp, matching exp08.generate_score."""

    def __init__(self, gguf_path: Path, device: str, threads: int, n_ctx: int = 2048):
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=str(gguf_path),
            n_ctx=n_ctx,
            n_gpu_layers=-1 if device == "gpu" else 0,
            n_threads=threads,
            n_threads_batch=threads,
            seed=0,
            verbose=False,
        )
        add_bos = str(self.llm.metadata.get("tokenizer.ggml.add_bos_token", "false")).lower()
        self.add_bos = add_bos == "true"

    def tokens(self, prompt: str) -> list[int]:
        ids = self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        bos = self.llm.token_bos()
        if self.add_bos and bos >= 0 and (not ids or ids[0] != bos):
            ids = [bos] + ids
        return ids

    def generate(self, prompt: str) -> str:
        # Reset so every answer is timed from a cold KV cache (no prefix reuse).
        self.llm.reset()
        out = self.llm.create_completion(
            self.tokens(prompt),
            max_tokens=32,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            min_p=0.0,
            repeat_penalty=1.0,
            seed=0,
        )
        return out["choices"][0]["text"]


def validate_file(gguf_path: Path, release: str, spec, processor, device: str,
                  threads: int, output_root: Path, hash_files: bool) -> None:
    import exp08_llm_finetune as exp08
    from validate_pintu_models import latency_summary, load_test_rows

    variant = variant_from_filename(gguf_path, release)
    tag = f"{variant}__{device}"
    with StageManifest("validate", release, tag,
                       args={"gguf": gguf_path, "device": device, "threads": threads},
                       hash_files=hash_files) as manifest:
        print("\n" + "=" * 72, flush=True)
        print(f"Validating: {release} {variant} on {device}", flush=True)
        print("=" * 72, flush=True)

        load_start = time.perf_counter()
        scorer = GGUFScorer(gguf_path, device, threads)
        load_seconds = time.perf_counter() - load_start

        test_df, test_rows_path = load_test_rows(spec)
        raw_scores, normalized, outputs, seconds = [], [], [], []
        parse_failures = 0
        for row in test_df.to_dict("records"):
            max_score = int(row["Max Score"])
            prompt = exp08.render_prompt_text(processor, row, with_answer=False)
            start = time.perf_counter()
            text = scorer.generate(prompt)
            seconds.append(time.perf_counter() - start)
            score, parsed = exp08.parse_score(text, max_score)
            parse_failures += int(not parsed)
            raw_scores.append(score)
            normalized.append(score / max(max_score, 1))
            outputs.append(text.strip())

        predictions = test_df.copy()
        predictions["pred_score"] = normalized
        predictions["pred_raw"] = raw_scores
        predictions["llm_raw_output"] = outputs
        metrics = exp08.llm_metrics(predictions)

        output_dir = output_root / release / tag
        output_dir.mkdir(parents=True, exist_ok=True)
        exp08.write_predictions(predictions, str(output_dir), "test")

        cost = {
            "device": device,
            "threads": threads,
            "load_seconds": float(load_seconds),
            "parse_failures": int(parse_failures),
            "weights_gb": round(gguf_path.stat().st_size / 1e9, 3),
            "peak_rss_gb": peak_rss_gb(),
            **latency_summary(seconds),
        }
        report = {
            "model": release,
            "variant": variant,
            "model_path": str(gguf_path),
            "runtime": "llama.cpp",
            "text_only": True,
            "input": "qar",
            "preprocess": "clean",
            "decoding": {"greedy": True, "max_new_tokens": 32},
            "test_rows_source": str(test_rows_path),
            "test_answers": len(test_df),
            "test_metrics": metrics,
            "cost": cost,
        }
        try:
            import llama_cpp

            report["llama_cpp_python"] = llama_cpp.__version__
        except Exception:
            pass
        with (output_dir / "validation.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, default=float)

        manifest.add_inputs(gguf_path, test_rows_path)
        manifest.add_outputs(output_dir)
        manifest.metrics.update({**metrics, **cost})
        print(json.dumps({**metrics, **cost}, indent=2, default=float), flush=True)

        del scorer
        gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-text-only", type=Path, metavar="FILE")
    parser.add_argument("--models", nargs="+", choices=("qwen", "gemma", "sealion"),
                        default=["qwen"])
    parser.add_argument("--gguf-root", type=Path, default=Path("publish/gguf"))
    parser.add_argument("--processor-root", type=Path, default=Path("publish/full_models"),
                        help="Merged model folders; their processor renders the chat template.")
    parser.add_argument("--output-root", type=Path, default=Path("publish/benchmark"))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1),
                        help="Fixed CPU thread count (logged for reproducible latency).")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Restrict to these quant types, e.g. Q8_0 Q4_K_M.")
    parser.add_argument("--no-hash", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if args.check_text_only:
        check_text_only(args.check_text_only)
        return

    from transformers import AutoProcessor
    from validate_pintu_models import SPECS

    for model_key in args.models:
        spec = SPECS[model_key]
        release = spec.release_name
        files = sorted((args.gguf_root / release).glob(f"{release}-*.gguf"))
        if args.only:
            wanted = {q.upper() for q in args.only}
            files = [f for f in files if parse_gguf_name(f, release)[0] in wanted]
        if not files:
            raise FileNotFoundError(f"No GGUF files for {release} in {args.gguf_root / release}")
        processor = AutoProcessor.from_pretrained(args.processor_root / release,
                                                  trust_remote_code=True)
        for gguf_path in files:
            validate_file(gguf_path, release, spec, processor, args.device,
                          args.threads, args.output_root, not args.no_hash)
    print("\nAll requested GGUF validations completed.", flush=True)


if __name__ == "__main__":
    np.random.seed(0)
    main()
