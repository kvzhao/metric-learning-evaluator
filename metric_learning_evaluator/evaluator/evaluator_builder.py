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

EVALUATION_DISPLAY_NAMES = {
    eval_fields.ranking: 'rank-eval',
    eval_fields.facenet: 'pair-eval',
}

def parse_results_to_tensorboard(dict_results):
    pass

class EvaluatorBuilder(object):
    """Evaluator Builder & Interface.
    """

    def __init__(self, embedding_size, logit_size, config_dict):
        """Evaluator Builder.

          The object builds evaluation functions according to the given configuration 
          and manage shared data (embeddings, labels and attributes) in container objects.

          Args:
            embedding_size, 

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
        from pprint import pprint
        pprint (config_dict)

        # allocate shared embedding containers
        container_size = self.configs.container_size
        self.embedding_size = embedding_size
        self.logit_size = logit_size

        self.embedding_container = EmbeddingContainer(embedding_size, logit_size, container_size)

        # TODO @kv: If no attributes are given, do not allocate it?
        self.attribute_container = AttributeContainer()

        self._build()

        # Allocate general query interface
        if not self.configs.database_type:
            # TODO @kv: consistent check with query condition
            self.query_interface = None
        else:
            self.query_interface = QueryInterface(self.configs.database_type)

    def _build(self):
        """
          Build:
            Parse the config and create evaluators.
            TODO @kv: Add a counter to calculate number of added data
        """

        # Parse the Configuration
        self.evaluations = {}
        for eval_name in self.configs.evaluation_names:
            if not eval_name in REGISTERED_EVALUATION_OBJECTS:
                print ('WARNING: {} is not registered would be skipped.'.format(eval_name))
                continue
            self.evaluations[eval_name] = REGISTERED_EVALUATION_OBJECTS[eval_name](self.configs)

    @property
    def evaluation_names(self):
        # NOTE: evaluation_types from config; evaluation_names from object instance.
        #return [_eval_name for _eval_name, _eval in self.evaluations.items()]
        return [_eval.evaluation_name for _, _eval in self.evaluations.items()]

    @property
    def metric_names(self):
        # TODO @kv: merge each evaluation names, metrics and attributes.
        _metric_names = []
        from pprint import pprint
        for _eval_name in self.evaluations:
            if _eval_name in EVALUATION_DISPLAY_NAMES:
                _eval_display_name = EVALUATION_DISPLAY_NAMES[_eval_name]
            else:
                _eval_display_name = _eval_name
            _metric_config = self.configs.get_per_eval_metrics(_eval_name)
            _attribute_list = self.configs.get_per_eval_attributes(_eval_name)
            pprint(_metric_config)
            pprint(_attribute_list)
            for _metric_type, _metric in _metric_config.items():
                if _metric is None or _attribute_list is None:
                    continue
                for _attr_name in _attribute_list:
                    for _metric_name, _metric_value in _metric.items():
                        print(_metric_name, _metric_value)
                        if _metric_value is None:
                            continue
                        name_string = '{}-{}-{}@{}'.format(_metric_type, _attr_name, _metric_name, _metric_value)
                        _metric_names.append(name_string)
        return _metric_names

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
            if not self.configs.required_attributes:
                print ('WARNING: No required attributes are pre-defined.')
            queried_attributes = self.query_interface.query(image_id, self.configs.required_attributes)
            self.attribute_container.add(image_id, queried_attributes)

    
    def evaluate(self):
        """Execute given evaluations and returns a dictionary of metrics.
        
          Return:
            total_metrics, dict:
        """

        total_metrics = {}

        #TODO: Pass containers when compute. (functional objects)
        #TODO @kv: Consider with metric_names together
        for _eval_name, _evaluation in self.evaluations.items():
            # Pass the container to the evaluation objects.
            print ('Execute {}'.format(_eval_name))
            per_eval_metrics = _evaluation.compute(self.embedding_container,
                                                   self.attribute_container)
            # TODO: flatten results and return
            total_metrics[_eval_name] = per_eval_metrics

        # Returned example:
        # {'RankingEvaluation': {'all_classes': {'top_1_hit_accuracy': {1: 0.9929824561403509},
        #                              'top_k_hit_accuracy': {5: 0.9976608187134502}}}}
        return total_metrics
        
    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        self.embedding_container.clear()
        self.attribute_container.clear()
