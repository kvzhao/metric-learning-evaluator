import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from metrics.multilabel_metrics import MultilabelMetrics


class TestMultilabelMetrics(unittest.TestCase):
    """Test Multilabel Metrics
    """
    preds = np.array([
        [1, 1],
        [1, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [1, 0],
        [0, 0],
        [0, 1]],
        dtype=bool)
    labels = np.array([
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 1],
        [0, 1],
        [0, 0],
        [0, 0],
        [0, 1]],
        dtype=bool)
    accuracy = 0.3125
    precision = 0.375
    recall = 0.3125
    f1_score = 1 / 3

    def test_initialize(self):
        metric_obj = MultilabelMetrics()
        self.assertTrue(metric_obj.is_empty)

    def test_add_inputs(self):
        metric_obj = MultilabelMetrics()
        metric_obj.add_inputs(self.preds, self.labels)
        self.assertFalse(metric_obj.is_empty)

        # invalid shape
        with self.assertRaises(ValueError):
            metric_obj = MultilabelMetrics()
            metric_obj.add_inputs(self.preds[:, np.newaxis], self.labels)

        # different shape
        with self.assertRaises(ValueError):
            metric_obj = MultilabelMetrics()
            metric_obj.add_inputs(self.preds[:2, :], self.labels)

        # invalid type
        with self.assertRaises(TypeError):
            metric_obj = MultilabelMetrics()
            metric_obj.add_inputs(self.preds.astype(np.int32), self.labels)

        # empty inputs
        with self.assertRaises(ValueError):
            metric_obj = MultilabelMetrics()
            metric_obj.add_inputs(self.preds[0:0, :], self.labels)

    def test_compute_metrics(self):
        metric_obj = MultilabelMetrics()
        metric_obj.add_inputs(self.preds, self.labels)
        self.assertAlmostEqual(metric_obj.accuracy, self.accuracy)
        self.assertAlmostEqual(metric_obj.precision, self.precision)
        self.assertAlmostEqual(metric_obj.recall, self.recall)
        self.assertAlmostEqual(metric_obj.f1_score, self.f1_score)


if __name__ == '__main__':
    unittest.main()
