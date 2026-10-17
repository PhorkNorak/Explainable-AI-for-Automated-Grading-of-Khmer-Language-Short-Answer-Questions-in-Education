"""Control-flow checks only: no training or subprocess is launched."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('audit_hpc', Path(__file__).resolve().parents[1]/'scripts/hpc_handoff.py')
hpc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hpc)


class HandoffTests(unittest.TestCase):
    def fixture(self, root, score=5):
        (root/'local').mkdir()
        (root/'local/audit_inputs.json').write_text('{}')
        (root/'local/data').mkdir()
        for name in ('dataset.csv', 'dataset_no_10c_biology.csv'):
            (root/'local/data'/name).write_text('Question,Reference,Answer,Student Score,Max Score\nq,r,a,'+str(score)+',15\n')
        (root/'local/results/results_other').mkdir(parents=True)
        (root/'local/results/results_other/old.json').write_text('{}')

    def test_unrelated_results_do_not_count_as_completed_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            with patch.object(hpc,'ROOT',root), patch.object(hpc.subprocess,'run',return_value=SimpleNamespace(returncode=0)):
                with self.assertRaises(RuntimeError): hpc.run('classical')
            self.assertFalse((root/'local/logs/audit_execution/classical.json').exists())

    def test_invalid_ground_truth_blocks_before_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root,19)
            with patch.object(hpc,'ROOT',root), patch.object(hpc.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
                with self.assertRaises(ValueError): hpc.run('classical')
            run.assert_not_called()

    def test_snapshot_copies_new_source_layout_without_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base/'repo'; root.mkdir()
            self.fixture(root)
            (root/'src/khmer_asag').mkdir(parents=True)
            (root/'src/khmer_asag/__init__.py').write_text('')
            (root/'pyproject.toml').write_text('[project]\nname="fixture"\n')
            with patch.object(hpc, 'ROOT', root):
                hpc.prepare(base/'snapshot')
            self.assertTrue((base/'snapshot/src/khmer_asag/__init__.py').is_file())
            self.assertTrue((base/'snapshot/local/data/dataset.csv').is_file())
            self.assertTrue((base/'snapshot/local/audit_inputs.json').is_file())
            self.assertFalse((base/'snapshot/local/results/results_other').exists())

    def test_receipt_hashes_outputs_in_private_results_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            with patch.object(hpc, 'ROOT', root):
                for folder in hpc.expected_runs('classical'):
                    folder.mkdir(parents=True)
                    for name in ('config.json', 'metrics.json', 'predictions_test.csv', 'predictions_val.csv'):
                        (folder/name).write_text('fixture')
                with patch.object(hpc.subprocess, 'run', return_value=SimpleNamespace(returncode=0)):
                    hpc.run('classical')
            receipt=json.loads((root/'local/logs/audit_execution/classical.json').read_text())
            self.assertTrue(receipt['artifacts'])
            self.assertTrue(all(p.startswith('local/results/') for p in receipt['artifacts']))

if __name__ == '__main__': unittest.main()
