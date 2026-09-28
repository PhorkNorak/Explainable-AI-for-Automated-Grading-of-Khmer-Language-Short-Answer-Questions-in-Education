"""Execute production handlers without app import (which trains on startup)."""
import ast
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import sys
import numpy as np
from xai.explainers import shap_importance

SOURCE = Path(__file__).resolve().parents[1]/'prototype/app.py'

def functions(*names, **context):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'np': np, 'os': os, 're': re, **context}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_LLM_ANSWER_MARKERS' for t in node.targets):
            namespace['_LLM_ANSWER_MARKERS'] = ast.literal_eval(node.value)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace

class PrototypeHandlerTests(unittest.TestCase):
    def setUp(self):
        self.grader = SimpleNamespace(name='fixture', preprocess_mode='clean', input_format='ra', score=Mock(return_value=.5))
        self.handlers = functions('grade', MODELS={'fixture':self.grader}, INACTIVE={},
                                  preprocess=lambda x,m:x, _feedback=lambda *a:'offline feedback')

    def test_invalid_maximum_does_not_call_model(self):
        for maximum in [0,-1,1.5,float('nan'),float('inf'),'bad']:
            output = self.handlers['grade']('', 'reference', 'answer', maximum, 'fixture', False)
            self.assertIn('positive integer', output[0])
        self.grader.score.assert_not_called()

    def test_nonfinite_model_output_has_no_grade(self):
        self.grader.score.return_value=float('nan')
        output=self.handlers['grade']('', 'reference','answer',10,'fixture',False)
        self.assertIn('could not grade',output[0])
        self.assertEqual(output[1:],('',''))

    def test_nearest_even_raw_score_and_disabled_explanation(self):
        result=self.handlers['grade']('', 'reference','answer',5,'fixture',False)
        self.assertIn('**2 / 5**',result[0])
        self.assertIn('Explanation off',result[1])

    def test_missing_model_reports_inactive(self):
        result=self.handlers['grade']('', 'reference','answer',5,'absent',False)
        self.assertIn('not active',result[0])

    def test_unparseable_endpoint_output_is_not_an_imputed_grade(self):
        parse=functions('_parse_llm_int')['_parse_llm_int']
        with self.assertRaises(ValueError): parse('No score available',10)

    def test_endpoint_requires_one_unambiguous_integer(self):
        parse=functions('_parse_llm_int')['_parse_llm_int']
        for text in ('-1', '3.5', '3/10', '3 or 4'):
            with self.subTest(text=text), self.assertRaises(ValueError): parse(text,10)
        self.assertEqual(parse('Score: 3',10),3)

    def test_absent_feedback_endpoint_returns_fallback_signal(self):
        feedback=functions('_llm_feedback', FEEDBACK_LLM_BASE_URL='')['_llm_feedback']
        self.assertIsNone(feedback('q','r','a',2,5,[]))

    def test_clean_answer_explanation_preserves_spaces(self):
        self.handlers.update(readable_tokens=lambda s:s.split(), shap_importance=shap_importance,
                             heatmap_html_fragment=lambda words,values,title:'rendered attribution')
        with patch.dict(sys.modules, {'khmernltk':SimpleNamespace(word_tokenize=lambda s:s.split())}):
            result=self.handlers['grade']('', 'reference','aa bb',5,'fixture',True)
        self.assertEqual(result[1],'rendered attribution')

if __name__ == '__main__': unittest.main()
