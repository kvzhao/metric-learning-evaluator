"""

    Requirement:
        pip install cmake
        pip install MulticoreTSNE
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from cycler import cycler
from matplotlib import pyplot as plt
from MulticoreTSNE import MulticoreTSNE

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer


class TSNE(object):

    def __init__(self,
                 container,
                 perplexity=30.0,
                 learning_rate=120.0,
                 n_componenets=2,
                 n_jobs=4,
                 n_iter=1000,
                 verbose=1000):
        """
          Args:
            container: EmbeddingContainer
        """
        self._container = container
        self._engine = MulticoreTSNE(
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_components=n_componenets,
            n_jobs=n_jobs,
            n_iter=n_iter,
            verbose=verbose)

        self._results = None
        self._ids = None
        self._label_ids = None
        self._label_names = None

    def run(self):
        """
          Return:
            result: 2D Numpy array with shape (N, 2)
        """
        ids = self._container.instance_ids
        features = self._container.get_embedding_by_instance_ids(ids)
        label_ids = self._container.get_label_by_instance_ids(ids)
        label_names = self._container.get_label_name_by_instance_ids(ids)
        results = self._engine.fit_transform(features, label_ids)
        self._results = results
        self._ids = np.asarray(ids)
        self._label_ids = np.asarray(label_ids)
        self._label_names = list(set(label_names))
        return results

    def save_fig(self, figure_path):
        if self._results is None:
            return
        classes = set(self._label_ids)
        fig, ax = plt.subplots()
        for n_class in classes:
            idx = self._label_ids == n_class
            ax.scatter(self._results[idx, 0], self._results[idx, 1],
                       label=n_class,
                       alpha=0.3,
                       edgecolors='none')
        ax.axis('off')
        ax.legend(self._label_names, numpoints=1, fancybox=True, shadow=True,
                  loc='best')
        fig.savefig(figure_path)
