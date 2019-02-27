"""
author: jeff
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from metric_learning_evaluator.metrics.metric_base import MetricBase


class ClassificationMetrics(MetricBase):
    """
    Classification Metrics, including
    1. true positive rate (TPR): recall of positive EXAMPLES
    2. false positive rate (FPR): recall error rate of negative EXAMPLES
    3. accuracy: precision of all EXAMPLES
    4. validation rate (VAL): precision of positive PREDICTIONS
    5. false accept rate (FAR): precision error rate of negative PREDICTIONS
    """
    epsilon = 1e-7

    def __init__(self):
        self.clear()

    @property
    def is_empty(self):
        return self._is_empty

    @property
    def true_positive_rate(self):
        return self._true_pos / (self._true_pos + self._false_neg + self.epsilon)

    @property
    def false_positive_rate(self):
        return self._false_pos / (self._false_pos + self._true_neg + self.epsilon)

    @property
    def accuracy(self):
        return (self._true_pos + self._true_neg) / self._num_preds

    @property
    def validation_rate(self):
        return self._true_pos / (self._num_pos + self.epsilon)

    @property
    def false_accept_rate(self):
        return self._false_neg / (self._num_neg + self.epsilon)

    def add_inputs(self, preds, labels):
        """
        Args:
            preds: an ndarray, of dtype np.bool
            labels: an ndarray, of dtype np.bool
        """
        if preds.shape != labels.shape:
            raise ValueError(
                'predictions and labels must be of same shape, but get preds: "{}" '
                'labels: "{}"'.format(preds.shape, labels.shape))
        if preds.dtype != bool or labels.dtype != bool:
            raise TypeError(
                'predictions and labels must be of np.bool dtype '
                'but get preds: "{}" and labels "{}" instead'
                .format(preds.dtype, labels.dtype))
        n_preds = len(preds)
        if n_preds == 0:
            raise ValueError('get empty prediction: {}'.format(preds))
        self._num_preds = n_preds
        self._preds = preds
        self._labels = labels
        tp = np.sum(np.logical_and(preds, labels))
        fp = np.sum(np.logical_and(preds, np.logical_not(labels)))
        tn = np.sum(np.logical_and(np.logical_not(preds), np.logical_not(labels)))
        fn = np.sum(np.logical_and(np.logical_not(preds), labels))
        self._true_pos = float(tp)
        self._false_pos = float(fp)
        self._true_neg = float(tn)
        self._false_neg = float(fn)
        self._num_pos = float(np.sum(preds))
        self._num_neg = self._num_preds - self._num_pos
        self._is_empty = False

    def clear(self):
        self._preds = None
        self._labels = None
        self._num_preds = None
        self._true_pos = None
        self._false_pos = None
        self._true_neg = None
        self._false_neg = None
        self._num_pos = None
        self._num_neg = None
        self._is_empty = True
