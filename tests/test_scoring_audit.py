import unittest
import numpy as np
from evaluate import metrics


class ScoringValidationTests(unittest.TestCase):
    def test_fractional_true_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics([0, 1], [0.5, 4])

    def test_nonfinite_predictions_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics([np.nan, 1], [0, 4])

    def test_raw_arrays_cannot_broadcast(self):
        with self.assertRaises(ValueError):
            metrics([0, 1], [0, 4], [10], [0, 10])

    def test_zero_maximum_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics([0, 1], [0, 4], [0, 10], [0, 10])

    def test_rounding_ties_and_five_class_macro_denominator(self):
        result = metrics([.125, .375, .625, .875], [0, 2, 2, 4], [4]*4, [0, 2, 2, 4])
        self.assertEqual(result['accuracy'], 1)
        self.assertEqual(result['f1_macro'], .6)
        self.assertEqual(result['raw_exact'], 1)


if __name__ == '__main__':
    unittest.main()
