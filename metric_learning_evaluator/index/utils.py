import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np

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

def angular_distance(vectorA, vectorB):
    """
      Args:
        vectorA : 1D numpy array
        vectorB : 1D numpy array
      Return:
        angle (degrees) between two vectors : Float
    """
    def _unit_vector(vector):
        return vector / np.linalg.norm(vector)

    v1_u = _unit_vector(vectorA)
    v2_u = _unit_vector(vectorB)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def indexing_array(distances, target_array):
    """Sort target array according to distances
      Args:
        distances:
        target_array:
      Returns:
        sorted_target
    """
    return target_array[distances.argsort()]

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

    for threshold in thresholds:
        distances = euclidean_distance(matrixA, matrixB)
        positive = np.less(distances, threshold)
        positives[threshold] = positive

    return positives
