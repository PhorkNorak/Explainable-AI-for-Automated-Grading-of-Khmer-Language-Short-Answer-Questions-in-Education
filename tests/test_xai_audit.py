import sys
import unittest
from unittest.mock import patch
import numpy as np
from xai.explainers import shap_importance
import ast
import json
import hashlib
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace


def exp09_helpers():
    path = Path(__file__).resolve().parents[1] / 'experiments/exp09_xai.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'_sample_test', '_save_sample_manifest', '_hash_files', '_save_attribution'}
    namespace = dict(np=np, json=json, hashlib=hashlib, os=os,
                     _ROOT=str(path.parents[1]), C=SimpleNamespace(SEED=42))
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class ManifestTests(unittest.TestCase):
    def test_rendering_describes_positive_shap_not_removal_effect(self):
        from xai.render_html import heatmap_html_fragment
        html = heatmap_html_fragment(['aa', 'bb'], np.array([.1, -.1]))
        self.assertIn('positive attribution', html)
        self.assertIn('negative', html)
        self.assertNotIn('removing this word', html)

    def test_sample_all_preserves_actual_indices(self):
        import pandas as pd
        df = pd.DataFrame({'score_label': [0, 1, 2]}, index=[10, 30, 50])
        self.assertEqual(exp09_helpers()['_sample_test'](df, 0), [10, 30, 50])

    def test_manifest_separates_source_identity_and_processed_input(self):
        import pandas as pd
        helpers = exp09_helpers()
        with tempfile.TemporaryDirectory() as tmp:
            frames = [pd.DataFrame({'Answer': ['aa bb'], 'Answer_proc': [answer]}) for answer in ['aa bb', 'aabb']]
            manifests = []
            for family, frame in zip(['classical', 'bilstm'], frames):
                helpers['_save_sample_manifest'](frame, [0], tmp, family, {}, max_evals=50, fraction=.2)
                manifests.append(json.loads(Path(tmp, family + '_sample_manifest.json').read_text()))
            self.assertEqual(manifests[0]['samples'][0]['raw_row_sha256'], manifests[1]['samples'][0]['raw_row_sha256'])
            self.assertNotEqual(manifests[0]['samples'][0]['processed_row_sha256'], manifests[1]['samples'][0]['processed_row_sha256'])
            self.assertEqual(manifests[0]['max_evals'], 50)
            self.assertEqual(manifests[0]['fraction'], .2)


class ShapAuditTests(unittest.TestCase):
    def test_budgeted_permutations_preserve_additive_contributions(self):
        calls = []
        def predict(answer, reference):
            calls.append(answer)
            return sum({'aa': .2, 'bb': .3, 'cc': .4}[w] for w in answer.split())
        with patch.dict(sys.modules, {'shap': None}):
            words, values = shap_importance(predict, 'aa bb cc', '', 'segment', max_evals=7)
        np.testing.assert_allclose(values, [.2, .3, .4])
        self.assertLessEqual(len(calls), 7)

    def test_impossible_budget_fails_before_predictions(self):
        calls = []
        with patch.dict(sys.modules, {'shap': None}), self.assertRaises(ValueError):
            shap_importance(lambda a, r: calls.append(a) or 0, 'aa bb cc', '', 'segment', max_evals=2)
        self.assertEqual(calls, [])

    def test_model_failure_is_not_silently_retried(self):
        def fail(a, r):
            raise RuntimeError('model unavailable')
        with self.assertRaisesRegex(RuntimeError, 'model unavailable'):
            shap_importance(fail, 'aa bb', '', 'segment')


if __name__ == '__main__':
    unittest.main()
