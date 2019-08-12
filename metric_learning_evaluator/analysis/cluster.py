import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from tqdm import tqdm
from metric_learning_evaluator.index.agent import IndexAgent

from metric_learning_evaluator.index.utils import euclidean_distance
from metric_learning_evaluator.index.utils import angular_distance
from metric_learning_evaluator.index.utils import indexing_array

from metric_learning_evaluator.metrics.ranking_metrics import RankingMetrics
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.utils.switcher import switch


class Cluster(object):
    """Cluster is a group of same label embeddings.
    """

    def __init__(self, instance_ids, embeddings, label_id,
        cluster_id=None, label_name=None):
        """
          Args:
            instance_ids: 1D numpy array
            embeddings: 2D numpy array
            label_id: An integer-like
            label_name: A string
        """
        self._instance_ids = instance_ids
        self._embeddings = embeddings
        assert len(self._embeddings.shape) == 2, 'embedding must be 2D'
        self._num_feat, self._dim_feat = self._embeddings.shape
        self._label_id = label_id
        self._cluster_id = cluster_id
        self._label_name = label_name
        # make sure len are equal

    def __repr__(self):
        pass

    @property
    def mean(self):
        return np.mean(self._embeddings, axis=0)

    @property
    def instance_ids(self):
        return self._instance_ids

    @property
    def center(self):
        """Need some algorithm
        """
        pass

    @property
    def features(self):
        # return all components inside the cluster
        return self._embeddings

    @property
    def area(self):
        """
          Compute cluster area
        """
        _center = self.mean
        _distances = np.sort(euclidean_distance(_center, self._embeddings))
        _rmax = _distances[-1] + 1e-8
        return _rmax ** 2

    @property
    def density(self):
        return self._num_feat / self.area

    @property
    def id(self):
        return self._cluster_id

    @property
    def label_name(self):
        return self._label_name