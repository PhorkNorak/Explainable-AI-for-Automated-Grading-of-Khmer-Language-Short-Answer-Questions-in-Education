import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from preprocess import preprocess
from xai.explainers import shap_importance

class PreprocessingTests(unittest.TestCase):
    def test_segmenter_failure_cannot_be_labelled_segment_mode(self):
        with patch.dict(sys.modules, {'khmernltk':None}), self.assertRaises(RuntimeError):
            preprocess('answer', 'segment')

    def test_clean_nfc_invisibles_punctuation_and_digits(self):
        self.assertEqual(preprocess('a\u0301\u200b។១2', 'clean'), 'á ១2')

    def test_shap_full_coalition_preserves_clean_whitespace(self):
        calls=[]
        def predict(a,r):
            calls.append(a)
            return 1.0 if a == 'aa bb' else 0.0
        with patch.dict(sys.modules, {'khmernltk':SimpleNamespace(word_tokenize=lambda t:['aa','bb'])}):
            _, phi=shap_importance(predict,'aa bb','','clean',max_evals=3)
        self.assertIn('aa bb',calls)
        self.assertAlmostEqual(sum(phi),1.)

if __name__ == '__main__': unittest.main()
