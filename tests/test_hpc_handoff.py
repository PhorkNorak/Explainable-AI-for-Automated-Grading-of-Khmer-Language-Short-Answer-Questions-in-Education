"""Control-flow checks only: no training or subprocess is launched."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('audit_hpc', Path(__file__).resolve().parents[1]/'docs/audit/hpc_handoff.py')
hpc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hpc)


class HandoffTests(unittest.TestCase):
    def fixture(self, root, score=5):
        (root/'audit_inputs.json').write_text('{}')
        (root/'data').mkdir()
        for name in ('dataset.csv', 'dataset_no_10c_biology.csv'):
            (root/'data'/name).write_text('Question,Reference,Answer,Student Score,Max Score\nq,r,a,'+str(score)+',15\n')
        (root/'results_other').mkdir()
        (root/'results_other/old.json').write_text('{}')

    def test_unrelated_results_do_not_count_as_completed_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            with patch.object(hpc,'ROOT',root), patch.object(hpc.subprocess,'run',return_value=SimpleNamespace(returncode=0)):
                with self.assertRaises(RuntimeError): hpc.run('classical')
            self.assertFalse((root/'audit_execution/classical.json').exists())

    def test_invalid_ground_truth_blocks_before_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root,19)
            with patch.object(hpc,'ROOT',root), patch.object(hpc.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
                with self.assertRaises(ValueError): hpc.run('classical')
            run.assert_not_called()

if __name__ == '__main__': unittest.main()
