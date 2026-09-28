"""CPU checks of LLM run metadata without loading model weights."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from experiments import exp08_llm_finetune as llm


class ZeroShotMetadataTests(unittest.TestCase):
    def test_ra_prediction_run_records_ra_input(self):
        # Model inference is deliberately substituted: this checks the real
        # configuration writer, not GPU generation.
        with tempfile.TemporaryDirectory() as tmp:
            llm._set_input_fmt("ra")
            try:
                with patch.object(llm.C, "RUNS_DIR", tmp), \
                     patch.object(llm, "load_model", return_value=(Mock(), Mock(), "test")), \
                     patch.object(llm.data, "apply_preprocess", return_value=None), \
                     patch.object(llm, "predict_split", return_value=None), \
                     patch.object(llm, "llm_metrics", return_value={"qwk": 0.0}), \
                     patch.object(llm, "write_predictions"):
                    llm.run_zeroshot("qwen35_4b", None, None, "ra_case", 1024)
                cfg = json.loads((Path(tmp) / "ra_case/config.json").read_text())
                self.assertEqual(cfg["input"], "ra")
            finally:
                llm._set_input_fmt("qar")


if __name__ == "__main__":
    unittest.main()
