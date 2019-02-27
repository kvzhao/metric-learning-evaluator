""" Distance functional object
  @kv
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy
from metrics.metric_base import MetricBase
from metrics.distances import batch_euclidean_distances

def euclidean_distance_filter():
    pass

class DistanceMetrics(MetricBase):
    """Distance Function Sets
      This object takes embeddings and a threshold, then
      user can extract indices which above the threshold.
      
    """

    epsilon = 1e-7

    def __init__(self):
        pass

    def euclidean_distance_filter(self, thresholds):
        pass