"""Quick smoke check: does Unsloth load the official Gemma 4 / Qwen 3.5 repos?

Runs in ~1 minute (downloads tokenizer + config, no weights to disk).
"""

# Support direct CLI execution from a source checkout.
import os as _layout_os
import sys as _layout_sys
_layout_root = _layout_os.path.dirname(_layout_os.path.dirname(_layout_os.path.abspath(__file__)))
for _layout_path in (_layout_root, _layout_os.path.join(_layout_root, "src")):
    if _layout_path not in _layout_sys.path:
        _layout_sys.path.insert(0, _layout_path)
from khmer_asag.paths import project_path

from unsloth import FastLanguageModel

CANDIDATES = [
    "google/gemma-4-E4B-it",
    "Qwen/Qwen3.5-4B",
]

for name in CANDIDATES:
    print(f"\n=== Trying {name} ===")
    try:
        model, tok = FastLanguageModel.from_pretrained(
            model_name=name,
            load_in_4bit=True,
            max_seq_length=128,
        )
        print(f"  UNSLOTH OK: {name}")
        del model, tok
    except Exception as e:
        msg = str(e)[:200]
        print(f"  UNSLOTH FAILED: {type(e).__name__}: {msg}")
        # Try plain transformers fallback
        try:
            from transformers import AutoConfig, AutoTokenizer
            cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            print(f"  HF FALLBACK OK: arch={cfg.architectures} vocab={tok.vocab_size}")
        except Exception as e2:
            print(f"  HF FALLBACK FAILED: {type(e2).__name__}: {str(e2)[:150]}")
