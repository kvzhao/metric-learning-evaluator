"""
    The evaluator builder for managing  customized evaluation metrics.

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

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer

from metric_learning_evaluator.query.general_database import QueryInterface
# import all evaluation objects
from metric_learning_evaluator.core.registered import REGISTERED_EVALUATION_OBJECTS
from metric_learning_evaluator.core.registered import EVALUATION_DISPLAY_NAMES
from metric_learning_evaluator.evaluations.evaluation_base import MetricEvaluationBase
from metric_learning_evaluator.evaluations.ranking_evaluation import RankingEvaluation
from metric_learning_evaluator.evaluations.facenet_evaluation import FacenetEvaluation
from metric_learning_evaluator.evaluations.classification_evaluation import ClassificationEvaluation
from metric_learning_evaluator.evaluations.geometric_evaluation import GeometricEvaluation
from metric_learning_evaluator.config_parser.parser import ConfigParser
from metric_learning_evaluator.core.standard_fields import ConfigStandardFields as config_fields
from metric_learning_evaluator.core.standard_fields import EvaluationStandardFields as eval_fields
from metric_learning_evaluator.core.standard_fields import AttributeStandardFields as attr_fields


class EvaluatorBuilder(object):
    """Evaluator Builder & Interface.
    """

    def __init__(self, embedding_size, prob_size, config_dict, mode='online'):
        """Evaluator Builder.

          The object builds evaluation functions according to the given configuration
          and manage shared data (embeddings, labels and attributes) in container objects.

          Args:
            embedding_size: Integer describes 1d embedding size.
            prob_size: Integer describes size of the logits.
            config_dict: Dict, loaded yaml foramt dict.
            mode: String, `online` or `offline`.

          Building procedure: (TODO @kv: update these steps)
            * parse the config
            * allocate containers
            * create evaluations
            * add datum
            * run evaluate
            * (optional) get update_ops
        TODO:
            - deprecate attribute container
        """
        self.configs = ConfigParser(config_dict)

        # allocate shared embedding containers
        container_size = self.configs.container_size
        self.embedding_size = embedding_size
        self.prob_size = prob_size

        self.embedding_container = EmbeddingContainer(embedding_size, prob_size, container_size)

        self.mode = mode
        if self.mode not in ['online', 'offline']:
            raise ValueError('Evaluator mode: {} is not defined.'.format(self.mode))

        self._build()

        self._instance_counter = 0
        self._total_metrics = {}
        self._results = {}
        # Allocate general query interface
        if not self.configs.database[config_fields.database_type]:
            # TODO @kv: consistent check with query condition
            print('No attribute database')
            self.query_interface = None
        else:
            self.query_interface = QueryInterface(self.configs.database)
            print('Attribute database is initialized.')

    def _build(self):
        """
          Build:
            Parse the config and create evaluators.
        """
        # Allocate evaluation object with corresponding configuration
        self.evaluations = {}
        for eval_name in self.configs.chosen_evaluation_names:
            if eval_name == eval_fields.classification and self.prob_size == 0:
                print('{} is assigned, but prob_size == 0, remove from the chosen list.'.format(eval_name))
                # remove the chosen name in the list
                self.configs.chosen_evaluation_names.remove(eval_name)
                continue
            eval_config = self.configs.get_eval_config(eval_name)
            self.evaluations[eval_name] = REGISTERED_EVALUATION_OBJECTS[eval_name](eval_config, self.mode)

    @property
    def evaluation_names(self):
        # NOTE: evaluation_types from config; evaluation_names from object instance.
        return self.configs.chosen_evaluation_names

    @property
    def metric_names(self):
        _metric_names = []
        for _eval_name in self.configs.chosen_evaluation_names:
            if _eval_name in EVALUATION_DISPLAY_NAMES:
                _display_eval_name = EVALUATION_DISPLAY_NAMES[_eval_name]
            else:
                _display_eval_name = _eval_name
            _metric_name_per_evaluation = self.evaluations[_eval_name].metric_names
            for _metric_name in _metric_name_per_evaluation:
                _metric_name = '{}/{}'.format(_display_eval_name, _metric_name)
                _metric_names.append(_metric_name)
        return _metric_names

    def add_instance_id_and_embedding(self, instance_id, label_id, embedding, probability=None):
        """Add embedding and label for a sample to be used for evaluation.

           If the query attribute names are given in config, this function will
           search them on database automatically.

        Args:
            instance_id, integer:
                A integer identifier for the image. instance_id
            label_id: An interger to describe class
            embedding, list or numpy array:
                Embedding, feature vector
        """

        # NOTE: If we call classification, then add probability.
        # TODO @kv: If instance_id is None, use index as default.
        if instance_id is None or instance_id == -1:
            instance_id = self._instance_counter

        if not isinstance(instance_id, int):
            instance_id = int(instance_id)
        if not isinstance(label_id, int):
            label_id = int(label_id)

        if self.query_interface:
            queried_attributes = self.query_interface.query(instance_id)
            self.embedding_container.add(instance_id, label_id,
                                         embedding, probability, attribute=queried_attributes)
        else:
            self.embedding_container.add(instance_id, label_id, embedding, probability)

        # verbose for developing stage.
        if self.embedding_container.counts % 1000 == 0:
            if probability is None:
                print('{} embeddings are added.'.format(self.embedding_container.counts))
            else:
                print('{} embeddings and probabilities are added.'.format(self.embedding_container.counts))

        self._instance_counter += 1

    def add_container(self, embedding_container=None):
        """Add filled containers which should be provided previously.

          Args:
            embedding_container: EmbeddingContainer, default is None.
          Notice:
            Sanity check:
          TODO @kv: Think about how to cooperate with attributes
        """
        # replace container
        if embedding_container is not None:
            if not isinstance(embedding_container, EmbeddingContainer):
                # raise error
                return
            self.embedding_container.clear()
            self.embedding_container = embedding_container
            print('Update embedding container.')

    def evaluate(self):
        """Execute given evaluations and returns a dictionary of metrics.
          Return:
            total_metrics: A flatten dictionary for display each measures
        """
        for _eval_name, _evaluation in self.evaluations.items():
            # Pass the container to the evaluation objects.
            res_container = _evaluation.compute(self.embedding_container)
            self._results[_eval_name] = res_container

            # TODO: flatten results and return
            if _eval_name in EVALUATION_DISPLAY_NAMES:
                _display_name = EVALUATION_DISPLAY_NAMES[_eval_name]
            else:
                _display_name = _eval_name
            if res_container:
                self._total_metrics[_display_name] = res_container.flatten
            else:
                self._total_metrics[_display_name] = {}
        flatten = {}
        for _eval_name, _content in self._total_metrics.items():
            for _metric, _value in _content.items():
                _combined_name = '{}/{}'.format(
                    _eval_name, _metric)
                flatten[_combined_name] = _value
        return flatten

    @property
    def results(self):
        return self._results

    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        self.embedding_container.clear()
        for _, _container in self._total_metrics.items():
            _container.clear()
