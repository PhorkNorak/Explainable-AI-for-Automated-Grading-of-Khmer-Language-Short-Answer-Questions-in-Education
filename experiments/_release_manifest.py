"""Provenance manifests for the Pintu release pipeline.

Every release stage (merge, validate, quantize, benchmark, upload) writes one JSON
manifest to ``publish/manifests/<stage>_<release>[_<variant>].json`` recording what
ran, on which code and hardware, with which inputs and outputs (SHA-256), and the
stage's own measurements. The release report and the Hub repos carry these files
so every published number traces back to a concrete run.

Python use::

    with StageManifest("merge", "Pintu-Qwen3.5-4B", args=vars(args)) as m:
        ...
        m.add_outputs(output_dir)
        m.metrics["n_targets"] = 196

Shell use (for the llama.cpp steps)::

    python experiments/_release_manifest.py record --stage quantize \
        --release Pintu-Qwen3.5-4B --variant gguf_q4_k_m \
        --inputs a.gguf --outputs b.gguf --extra llama_cpp_commit=abc123
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = Path("publish/manifests")

TRACKED_PACKAGES = (
    "torch", "transformers", "peft", "bitsandbytes", "accelerate", "unsloth",
    "safetensors", "huggingface_hub", "llama_cpp_python", "gguf", "numpy",
    "pandas", "scikit-learn",
)

# Hash files up to this size fully; larger weight files are hashed too (they are
# what gets published), but callers may disable hashing for quick dry runs.
_CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_files(paths: Iterable[Path], hash_files: bool = True) -> list[dict]:
    """List files (directories are expanded) with size and optional SHA-256."""
    rows: list[dict] = []
    for path in paths:
        path = Path(path)
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for file in files:
            if not file.exists():
                rows.append({"path": str(file), "missing": True})
                continue
            row = {"path": str(file), "bytes": file.stat().st_size}
            if hash_files:
                row["sha256"] = sha256_file(file)
            rows.append(row)
    return rows


def git_state() -> dict:
    def run(*cmd: str) -> str:
        try:
            return subprocess.run(
                cmd, cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return ""

    commit = run("git", "rev-parse", "HEAD")
    dirty = run("git", "status", "--porcelain")
    return {"commit": commit or None, "dirty": bool(dirty) if commit else None}


def package_versions() -> dict:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


def peak_rss_gb() -> float | None:
    """Peak resident memory of this process in GB (None if unavailable)."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS bytes.
        return round(peak / (1e9 if sys.platform == "darwin" else 1e6), 3)
    except Exception:
        pass
    try:
        import psutil

        mem = psutil.Process().memory_info()
        return round(getattr(mem, "peak_wset", mem.rss) / 1e9, 3)
    except Exception:
        return None


def manifest_path(stage: str, release: str, variant: str | None = None,
                  root: Path = MANIFEST_DIR) -> Path:
    name = f"{stage}_{release}" + (f"_{variant}" if variant else "")
    return Path(root) / f"{name}.json"


class StageManifest:
    """Context manager that times a stage and writes its manifest on exit.

    The manifest is written even when the stage fails, with ``status`` set to
    ``failed`` and the error message, so failed attempts stay visible.
    """

    def __init__(self, stage: str, release: str, variant: str | None = None,
                 args: dict | None = None, root: Path = MANIFEST_DIR,
                 hash_files: bool = True):
        self.stage, self.release, self.variant = stage, release, variant
        self.args = {k: str(v) if isinstance(v, Path) else v for k, v in (args or {}).items()}
        self.root, self.hash_files = Path(root), hash_files
        self.inputs: list[dict] = []
        self.outputs: list[dict] = []
        self.metrics: dict = {}
        self.extra: dict = {}
        self.path = manifest_path(stage, release, variant, self.root)

    def add_inputs(self, *paths: Path) -> None:
        self.inputs.extend(describe_files(paths, self.hash_files))

    def add_outputs(self, *paths: Path) -> None:
        self.outputs.extend(describe_files(paths, self.hash_files))

    def __enter__(self) -> "StageManifest":
        self._t0 = time.time()
        self._started = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        record = {
            "stage": self.stage,
            "release": self.release,
            "variant": self.variant,
            "status": "failed" if exc_type else "ok",
            "error": f"{exc_type.__name__}: {exc}" if exc_type else None,
            "started_utc": self._started,
            "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "wall_seconds": round(time.time() - self._t0, 2),
            "command": " ".join(sys.argv),
            "args": self.args,
            "git": git_state(),
            "packages": package_versions(),
            "hardware": hardware(),
            "peak_rss_gb": peak_rss_gb(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metrics": self.metrics,
            "extra": self.extra,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, default=str)
        print(f"[manifest] {self.path}", flush=True)
        return False  # never swallow the stage's exception


def _parse_extra(pairs: list[str]) -> dict:
    extra = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        extra[key] = value
    return extra


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a release-stage manifest.")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--stage", required=True)
    rec.add_argument("--release", required=True)
    rec.add_argument("--variant")
    rec.add_argument("--inputs", nargs="*", default=[])
    rec.add_argument("--outputs", nargs="*", default=[])
    rec.add_argument("--extra", nargs="*", default=[], help="key=value pairs")
    rec.add_argument("--wall-seconds", type=float)
    rec.add_argument("--no-hash", action="store_true")
    rec.add_argument("--root", type=Path, default=MANIFEST_DIR)
    args = parser.parse_args()

    manifest = StageManifest(args.stage, args.release, args.variant,
                             args={"source": "shell"}, root=args.root,
                             hash_files=not args.no_hash)
    with manifest:
        manifest.add_inputs(*map(Path, args.inputs))
        manifest.add_outputs(*map(Path, args.outputs))
        manifest.extra.update(_parse_extra(args.extra))
    if args.wall_seconds is not None:
        data = json.loads(manifest.path.read_text(encoding="utf-8"))
        data["wall_seconds"] = args.wall_seconds
        manifest.path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                 encoding="utf-8")


if __name__ == "__main__":
    main()
