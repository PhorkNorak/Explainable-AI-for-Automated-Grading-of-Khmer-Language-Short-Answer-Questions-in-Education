"""Layout contracts: imports and private artifact locations, without model loading."""
from pathlib import Path
import tempfile
import unittest

from khmer_asag import paths


class LayoutTests(unittest.TestCase):
    def test_data_and_results_are_private(self):
        self.assertEqual(Path(paths.project_path('data', 'dataset.csv')), paths.LOCAL_ROOT / 'data/dataset.csv')
        self.assertEqual(Path(paths.project_path('results_no10c', 'runs')), paths.LOCAL_ROOT / 'results/results_no10c/runs')

    def test_champions_and_logs_are_private(self):
        self.assertEqual(Path(paths.project_path('results', 'champions')), paths.LOCAL_ROOT / 'checkpoints/results/champions')
        self.assertEqual(Path(paths.project_path('logs', 'run.log')), paths.LOCAL_ROOT / 'logs/run.log')

    def test_absolute_output_override_is_preserved(self):
        absolute = Path(tempfile.gettempdir()).resolve() / 'explicit-output'
        self.assertEqual(Path(paths.project_path(str(absolute), 'metrics.json')), absolute / 'metrics.json')

    def test_source_and_artifacts_have_separate_roots(self):
        self.assertEqual(paths.PACKAGE_ROOT, paths.REPOSITORY_ROOT / 'src/khmer_asag')
        self.assertEqual(paths.LOCAL_ROOT, paths.REPOSITORY_ROOT / 'local')

    def test_source_provenance_paths_resolve_to_package(self):
        for name in ['config.py', 'data.py', 'preprocess.py', 'evaluate.py', 'train.py', 'xai/explainers.py']:
            self.assertTrue(Path(paths.project_path(name)).is_file(), name)


if __name__ == '__main__':
    unittest.main()
