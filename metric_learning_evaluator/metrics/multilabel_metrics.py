"""
author: jeff
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from metric_learning_evaluator.metrics.metric_base import MetricBase


class MultilabelMetrics(MetricBase):
    """Multi-label Metrics, including:
    1. accuracy: mean IoU of predictions and labels
    2. precision: mean precision
    3. recall: mean recall
    4. f1-score: mean f1-score

    reference: https://stats.stackexchange.com/questions/12702/what-are-the-measure-for-accuracy-of-multilabel-data
    """
    epsilon = 1e-7

    def __init__(self):
        self._is_empty = True
        self.clear()

    @property
    def is_empty(self):
        return self._is_empty

    @property
    def accuracy(self):
        acc_arr = self._intersection / (self._union + self.epsilon)
        return float(np.sum(acc_arr)) / (self._num_preds + self.epsilon)

    @property
    def precision(self):
        precision_arr = self._intersection / (self._preds_n_true + self.epsilon)
        return float(np.sum(precision_arr)) / (self._num_preds + self.epsilon)

    @property
    def recall(self):
        recall_arr = self._intersection / (self._labels_n_true + self.epsilon)
        return float(np.sum(recall_arr)) / (self._num_preds + self.epsilon)

    @property
    def f1_score(self):
        f1_arr = 2 * self._intersection / (self._labels_n_true + self._preds_n_true + self.epsilon)
        return float(np.sum(f1_arr)) / (self._num_preds + self.epsilon)

    def add_inputs(self, preds, labels):
        """Add N input pairs with C classes output each pair

        Args:
            preds: an ndarray, of dtype np.bool, shape (N, C)
            labels: an ndarray, of dtype np.bool, shape (N, C)
        """
        if len(preds.shape) != 2 or len(labels.shape) != 2:
            raise ValueError(
                'shape must be (N, C), but get {preds} (preds), {labels} (labels)'.format(
                    preds=preds.shape,
                    labels=labels.shape))
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
        self._intersection = np.sum(np.logical_and(preds, labels), axis=1).astype(np.float32)
        self._union = np.sum(np.logical_or(preds, labels), axis=1).astype(np.float32)
        self._preds_n_true = np.sum(preds, axis=1).astype(np.float32)
        self._labels_n_true = np.sum(labels, axis=1).astype(np.float32)
        self._is_empty = False

    def clear(self):
        self._num_preds = None
        self._preds = None
        self._labels = None
        self._intersection = None
        self._union = None
        self._preds_n_true = None
        self._labels_n_true = None
        self._is_empty = True
