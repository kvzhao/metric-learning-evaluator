
"""
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import euclidean_distances

def pairwise_euclidean_distance(vector_1, vector_2):
    """Calculate the distance between two vectors.
      Return:
        dist, float
    """
    return euclidean_distances(vector_1, vector_2)


def batch_euclidean_distances(vector, matrix, p=2):
    """Calculate Euclidean distance between one vector and bunch of vectors
       (matrix with same .. dimension).
      Args:
        vector, 1D numpy array with shape (vec_dim)
        matrix, 2D numpy array with shape (batch_size, vec_dim)
        p, float: Order of the exponent of distance in flat space (Minkowski p-norm)
            1 <= p <= infinity
            p = 1: L1 norm
            p = 2: L2 norm
            p = np.inf: Infinite norm
      Return:
        distances, 1D (batch_size) array of distances
    """

    if isinstance(vector, list):
        vector = np.asarray([vector])

    if isinstance(vector, (np.generic, np.ndarray)) and len(vector.shape) == 1:
        vector = np.expand_dims(vector, axis=0)

    assert(vector.shape[0] == matrix.shape[1], 'Dimension of the row should match.')

    distances = np.squeeze(distance_matrix(vector, matrix, p), axis=0)

    return distances

def batch_some_distances(vector, matrix):
    # implement for testing
    pass