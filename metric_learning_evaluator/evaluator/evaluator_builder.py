"""
    The evaluator builder for managing  customized evaluation metircs.

    EvaluatorBuilder
        - Data structure
        - Computation combination

    Example usage of the Evaluator:
    ------------------------------
    evaluator = EvaluatorBuilder(config)

    # Embedding and label for image 1.
    evaluator.add_single_embedding_and_label(...)

    # Embedding and label for image 2.
    evaluator.add_single_embedding_and_label(...)

    metrics_dict = evaluator.evaluate()

    @bird, kv
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))  # noqa

import yaml
import numpy as np


from metric_learning_evaluator.evaluator.data_container import EmbeddingContainer
from metric_learning_evaluator.evaluator.data_container import AttributeContainer
from metric_learning_evaluator.evaluator.evaluation_base import MetricEvaluationBase

from metric_learning_evaluator.core.eval_standard_fields import ConfigStandardFields as config_fields
from metric_learning_evaluator.core.eval_standard_fields import AttributeStandardFields as attr_fields
from metric_learning_evaluator.core.eval_standard_fields import EvaluationStandardFields as eval_fields
from metric_learning_evaluator.core.config_parser import ConfigParser

from metric_learning_evaluator.query.general_database import QueryInterface

# import all evaluation objects
from metric_learning_evaluator.evaluator.classification_evaluation import ClassificationEvaluation
from metric_learning_evaluator.evaluator.mock_evaluation import MockEvaluation

from metric_learning_evaluator.evaluator.ranking_evaluation import RankingEvaluation
from metric_learning_evaluator.evaluator.facenet_evaluation import FacenetEvaluation

# registered evaluation objects
REGISTERED_EVALUATION_OBJECTS = {
    eval_fields.mock: MockEvaluation,
    eval_fields.classification: ClassificationEvaluation,
    eval_fields.ranking: RankingEvaluation,
    eval_fields.facenet: FacenetEvaluation,
}

def parse_results_to_tensorboard(dict_results):
    pass


class EvaluatorBuilder(object):
    """Evaluator Builder & Interface.
    """

    def __init__(self, config_path):
        """Evaluator Builder.

          The object builds evaluation functions according to the given configuration 
          and manage shared data (embeddings, labels and attributes) in container objects.

          Building procedure: (TODO @kv: update these steps)
            * parse the config
            * allocate containers
            * create evaluations
            * add datum
            * run evaluate
            * (optional) get update_ops
        """

        self.config = ConfigParser(config_path)

        # allocate shared embedding containers
        embedding_size = self.config.embedding_size
        container_size = self.config.container_size
        logit_size = self.config.logit_size

        self.embedding_container = EmbeddingContainer(embedding_size, logit_size, container_size)

        # TODO @kv: If no attributes are given, do not allocate it?
        self.attribute_container = AttributeContainer()

        self._build()

        # Allocate general query interface
        if not self.config.database_type:
            # TODO @kv: consistent check with query condition
            self.query_interface = None
        else:
            self.query_interface = QueryInterface(self.config.database_type)

    def _build(self):
        """
          Build:
            Parse the config and create evaluators.
            TODO @kv: Add a counter to calculate number of added data
        """

        # Parse the Configuration
        self.evaluations = {}
        for eval_name in self.config.evaluation_names:
            if not eval_name in REGISTERED_EVALUATION_OBJECTS:
                print ('WARNING: {} is not registered would be skipped.'.format(eval_name))
                continue
            self.evaluations[eval_name] = REGISTERED_EVALUATION_OBJECTS[eval_name](self.config)

    @property
    def evaluation_names(self):
        # NOTE: evaluation_types from config; evaluation_names from object instance.
        return [_eval.evaluation_name for _, _eval in self.evaluations.items()]

    def add_image_id_and_embedding(self, image_id, label_id, embedding, logit=None):
        """Add embedding and label for a sample to be used for evaluation.
           If the query attribute names are given in config, this function will
           search them on database automatically.

        Args:
            image_id, integer:
                A integer identifier for the image.
            label_id, integer:
                An index of label.
            embedding, list or numpy array:
                Embedding, feature vector
        """

        # NOTE: If we call classification, then add logit.
        # TODO @kv: If image_id is None, use index as default.
        self.embedding_container.add(image_id, label_id, embedding, logit)
        # verbose for developing stage.
        if self.embedding_container.counts % 1000 == 0:
            print ('{} embeddings are added.'.format(self.embedding_container.counts))
        
        # TODO: consider move `add_image_id_and_query_attribute` here.
        # Collect all `attribute_name`
        # if not evaluations_need_query: no queries are needed.

        if self.query_interface:
            if not self.config.required_attributes:
                print ('WARNING: No required attributes are pre-defined.')
            queried_attributes = self.query_interface.query(image_id, self.config.required_attributes)
            self.attribute_container.add(image_id, queried_attributes)

    
    def evaluate(self):
        """Execute given evaluations and returns a dictionary of metrics.
        
          Return:
            total_metrics, dict:
        """

        total_metrics = {}

        #TODO: Pass containers when compute. (functional objects)

        for _eval_name, _evaluation in self.evaluations.items():
            # Pass the container to the evaluation objects.
            print ('Execute {}'.format(_eval_name))
            per_eval_metrics = _evaluation.compute(self.embedding_container,
                                                   self.attribute_container)
            total_metrics[_eval_name] = per_eval_metrics.results

        return total_metrics


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
        def update_op(image_ids_batched,):
            pass

        #update_op = tf.py_func()

        def first_value_func():
            pass
        def value_func_factory(metric_name):
            def value_func():
                pass
            return value_func
        
    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        self.embedding_container.clear()
        self.attribute_container.clear()
