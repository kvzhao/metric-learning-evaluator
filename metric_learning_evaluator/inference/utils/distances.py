
"""Functions used for calculating distances.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))  # noqa

import numpy as np

class DistanceFunctionStandardFields:
    euclidean = 'euclidean'


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

    if truncation:
        sorted_target = target_array[distances.argsort()[:truncation]]
    else:
        sorted_target = target_array[distances.argsort()]

    return sorted_target


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