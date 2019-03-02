"""
author: jeff
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from metric_learning_evaluator.metrics.metric_base import MetricBase


class RankingMetrics(MetricBase):
    """
    Rank metrics class, supporting 4 kinds of metric.
    The metrics is evaluated on rank lists of m queries.
    1. top-1 hit accuracy: the average number of top-1 success recall in rank lists
    2. top-k hit accuracy: the average number of top-k success recall in rank lists
    3. average precisions: the average precision of each rank lists
    4. mean average precision: mean of all m average precisions
    """
    epsilon = 1e-7

    def __init__(self, top_k):
        self._top_k = top_k
        self.clear()

    @property
    def is_empty(self):
        return self._is_empty

    @property
    def top_k(self):
        return self._top_k

    @property
    def top1_hit_accuracy(self):
        return np.mean(self._hit_arrays[:, 0])

    @property
    def topk_hit_accuracy(self):
        return np.mean(np.any(self._hit_arrays[:, :self._top_k], axis=1))

    @property
    def average_precisions(self):
        return self._average_precisions

    @property
    def mean_average_precision(self):
        return np.mean(self._average_precisions)

    def add_inputs(self, hit_arrays):
        """
        Args:
            hit_arrays: a ndarray, of shape (K, N) and dtype bool.
                hit_arr[i][j] is true if the label of j-th item in rank list
                matches the label of i-th query
        """
        if len(hit_arrays.shape) != 2:
            raise ValueError(
                'hit array must be rank 2, but get "{}" instead'
                .format(len(hit_arrays.shape)))
        if not np.issubdtype(np.bool, hit_arrays.dtype):
            raise TypeError(
                'hit array must be of dtype bool/np.bool, but get "{}" instead'
                .format(hit_arrays.dtype))
        if hit_arrays.shape[0] == 0 or hit_arrays.shape[1] == 0:
            raise ValueError(
                'hit array must not be empty, but get shape "{}"'
                .format(hit_arrays.shape))
        self._hit_arrays = hit_arrays
        self._average_precisions = np.zeros((hit_arrays.shape[0], 1))
        for i, hit_arr in enumerate(hit_arrays):
            # NOTE @jeff:
            # AP measures the quality (precision) when recalling positive examples, i.e.
            # , the average number of attempts at each successful recall.
            # Therefore:
            # number of attempts = recall indexes + 1 (0 if no successful recall)
            # AP = number of attempts / number of successful recall
            #
            # The benefits of "0 AP for 0 successful hit" are 3-folded:
            # 1. it can be aggregated to mAP unlike NaN/None.
            # 2. if it results in 0 mAP, we can still know something wrong with the model
            # 3. it's reasonable to assign 0 mAP to total failure since total success get 1 mAP
            indices = np.where(hit_arr == 1)[0]
            average_precision = 0.
            n_pos = len(indices)
            if n_pos > 0:
                average_precision = np.mean((np.arange(n_pos) + 1) / (indices + 1))
            self._average_precisions[i] = average_precision
        self._is_empty = False

    def clear(self):
        self._hit_arrays = None
        self._average_precisions = None
        self._is_empty = True
