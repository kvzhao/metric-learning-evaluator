
"""Functions used for calculating distances.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))  # noqa

import numpy as np


def indexing_array(distances, target_array, truncation=None):
    """Sort target array according to distances
      Args:
        distances:
        target_array:
        truncation:
            integer or None, the end of array index
      Returns:
        sorted_target
    """

    # Check type and size

    if truncation:
        sorted_target = target_array[distances.argsort()][:truncation]
    else:
        sorted_target = target_array[distances.argsort()]

    return sorted_target


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

    assert(vector.shape[0] == matrix.shape[1]), 'Dimension of the row should match.'
    distances = np.squeeze(distance_matrix(vector, matrix, p), axis=0)
    return distances

def euclidean_distance(matrixA, matrixB):
    """
      Args:
        matrixA: 2D numpy array
        matrixB: 2D numpy array
      Return:
        distances: 
    """
    if not isinstance(matrixA, (np.generic, np.ndarray)):
        matrixA = np.asarray(matrixA)
    if not isinstance(matrixB, (np.generic, np.ndarray)):
        matrixB = np.asarray(matrixB)

    if not matrixA.shape[0] == matrixB.shape[0]:
        print ("")
    
    distances = np.sum(np.square(np.subtract(matrixA, matrixB)), axis=1)

    return distances

def euclidean_distance_filter(matrixA, matrixB, thresholds=[0.5, 1.0]):
    """
      Args:
        matrixA: 2d numpy array
        matrixB
      Return:
        positives: A dictionary of list

      NOTE:
        Should we use normalized distance?
    """
    positives = {}

    if isinstance(thresholds, float):
        thresholds = [thresholds]
    else:
        # as some iterable
        pass

    for threshold in thresholds:
        distances = euclidean_distance(matrixA, matrixB)
        positive = np.less(distances, threshold)
        positives[threshold] = positive

    return positives


def cosine_distance():
    pass


def consine_distance_filter():
    pass