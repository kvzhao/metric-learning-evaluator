from abc import ABCMeta
from abc import abstractmethod
import collections
import logging
import numpy as np


class EvaluationMetic(object):    
    """Interface for object detection evalution classes.

    Example usage of the Evaluator:
    ------------------------------
    evaluator = DetectionEvaluator()

    # Embedding and label for image 1.
    evaluator.add_single_embedding_and_label(...)

    # Embedding and laebl for image 2.
    evaluator.add_single_embedding_and_label(...)

    metrics_dict = evaluator.evaluate()
    """
    __metaclass__ = ABCMeta

    def __init__(self, embedding_dim):
        """Constructor.
        """
        self._embedding_list = []
        self._label_list = []
        # store all image ids for each category 
        self._cate_id_set = collections.OrderedDict([])

    @abstractmethod
    def add_single_embedding_and_label(self, embedding, label):
        """Adds embadding and label for a sample to be used for evaluation.

        Args:
            embedding: A float numpy array generated by network.
            label: A integer identitfier for the image.
        """
        pass
    
    @abstractmethod
    def get_estimator_eval_metric_ops(self, eval_dict): 
        """Returns dict of metrics to use with `tf.estimator.EstimatorSpec`.

        Note that this must only be implemented if performing evaluation with a
        `tf.estimator.Estimator`.

        Args:
          eval_dict: A dictionary that holds tensors for evaluating an object
            detection model, returned from
            eval_util.result_dict_for_single_example().

        Returns:
          A dictionary of metric names to tuple of value_op and update_op that can
          be used as eval metric ops in `tf.estimator.EstimatorSpec`. 
        """
        pass
    
    @abstractmethod
    def evaluate(self):
        """Evaluates detections and returns a dictionary of metrics."""
        pass

    @abstractmethod
    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        pass
