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

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.data_tools.attribute_container import AttributeContainer

from metric_learning_evaluator.query.general_database import QueryInterface
from metric_learning_evaluator.query.standard_fields import AttributeStandardFields as attr_fields

# import all evaluation objects
from metric_learning_evaluator.core.registered import REGISTERED_EVALUATION_OBJECTS
from metric_learning_evaluator.core.registered import EVALUATION_DISPLAY_NAMES

from metric_learning_evaluator.evaluations.standard_fields import EvaluationStandardFields as eval_fields
from metric_learning_evaluator.evaluations.evaluation_base import MetricEvaluationBase
from metric_learning_evaluator.evaluations.ranking_evaluation import RankingEvaluation
from metric_learning_evaluator.evaluations.facenet_evaluation import FacenetEvaluation
from metric_learning_evaluator.evaluations.checkout_evaluation import CheckoutEvaluation

from metric_learning_evaluator.config_parser.standard_fields import ConfigStandardFields as config_fields
from metric_learning_evaluator.config_parser.parser import ConfigParser


class EvaluatorBuilder(object):
    """Evaluator Builder & Interface.
    """

    def __init__(self, embedding_size, logit_size, config_dict, mode='online'):
        """Evaluator Builder.

          The object builds evaluation functions according to the given configuration 
          and manage shared data (embeddings, labels and attributes) in container objects.

          Args:
            embedding_size: Integer describes 1d embedding size.
            logit_size:
            config_dict: Dict, loaded yaml foramt dict.
            mode: String, `online` or `offline`.

          Building procedure: (TODO @kv: update these steps)
            * parse the config
            * allocate containers
            * create evaluations
            * add datum
            * run evaluate
            * (optional) get update_ops
        """

        # TODO @kv: Change config_path to parsed dictionary
        self.configs = ConfigParser(config_dict)

        # allocate shared embedding containers
        container_size = self.configs.container_size
        self.embedding_size = embedding_size
        self.logit_size = logit_size

        self.embedding_container = EmbeddingContainer(embedding_size, logit_size, container_size)
        self.attribute_container = AttributeContainer()

        self.mode = mode
        if self.mode not in ['online', 'offline']:
            raise ValueError('Evaluator mode: {} is not defined.'.format(self.mode))

        self._build()

        self._instance_counter = 0
        self._total_metrics = {}

        # Allocate general query interface
        if not self.configs.database:
            # TODO @kv: consistent check with query condition
            self.query_interface = None
        else:
            self.query_interface = QueryInterface(self.configs.database)

    def _build(self):
        """
          Build:
            Parse the config and create evaluators.
        """

        # Parse the Configuration
        self.evaluations = {}
        for eval_name in self.configs.chosen_evaluation_names:
            self.evaluations[eval_name] = REGISTERED_EVALUATION_OBJECTS[eval_name](self.configs)

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

    def add_instance_id_and_embedding(self, instance_id, label_id, embedding, logit=None):
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

        # NOTE: If we call classification, then add logit.
        # TODO @kv: If instance_id is None, use index as default.
        if instance_id is None or instance_id == -1:
            instance_id = self._instance_counter
        self.embedding_container.add(instance_id, label_id, embedding, logit)
        # verbose for developing stage.
        if self.embedding_container.counts % 1000 == 0:
            print ('{} embeddings are added.'.format(self.embedding_container.counts))
        
        # TODO: consider move `add_instance_id_and_query_attribute` here.
        # Collect all `attribute_name`
        # if not evaluations_need_query: no queries are needed.

        if self.query_interface:
            # TODO @dennis.liu : use grouping rules instead required_attributes
            # if not self.configs.required_attributes:
            #     print ('WARNING: No required attributes are pre-defined.')
            queried_attributes = self.query_interface.query(instance_id)
            self.attribute_container.add(int(instance_id), queried_attributes)

        self._instance_counter += 1
    
    def evaluate(self):
        """Execute given evaluations and returns a dictionary of metrics.
        
          Return:
            total_metrics, dict:
        """

        #TODO: Pass containers when compute. (functional objects)
        #TODO @kv: Consider with metric_names together
        for _eval_name, _evaluation in self.evaluations.items():
            # Pass the container to the evaluation objects.
            res_container = _evaluation.compute(self.embedding_container,
                                                   self.attribute_container)
            # TODO: flatten results and return
            if _eval_name in EVALUATION_DISPLAY_NAMES:
                _display_name = EVALUATION_DISPLAY_NAMES[_eval_name]
            else:
                _display_name = _eval_name
            self._total_metrics[_display_name] = res_container.flatten

        # Return example:
        # dict = {'RankingEvaluation': {'all_classes': {'top_1_hit_accuracy': {1: 0.9929824561403509},
        #                              'top_k_hit_accuracy': {5: 0.9976608187134502}}}}
        # convert to
        # {'ranking-all_classes-top_k_hit_accuracy@1': 0.9 ,}

        flatten = {}
        for _eval_name, _content in self._total_metrics.items():
            for _metric, _value in _content.items():
                _combined_name = '{}/{}'.format(
                    _eval_name, _metric)
                flatten[_combined_name] = _value
        
        return flatten
        
    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        self.embedding_container.clear()
        self.attribute_container.clear()

        for _, _container in self._total_metrics.items():
            _container.clear()
