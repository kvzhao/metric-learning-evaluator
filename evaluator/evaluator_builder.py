"""
    The evaluator builder for managing  customized evaluation metircs.

    EvaluatorBuilder
        - Data structure
        - Computation combination

    Example usage of the Evaluator:
    ------------------------------
    evaluator = EvaluatorBuilder(eval_config)

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
    os.path.join(os.path.dirname(__file__), '..')))

import yaml
import numpy as np

from evaluator.evaluation_base import EmbeddingContainer
from evaluator.evaluation_base import AttributeContainer
from evaluator.evaluation_base import MetricEvaluationBase

from core.eval_standard_fields import EvalConfigStandardFields as config_fields

from query.general_database import QueryInterface

# import all evaluation objects
from evaluator.classification_evaluation import ClassificationEvaluation
from evaluator.mock_evaluation import MockEvaluation

from evaluator.ranking_evaluation import RankingEvaluation

# registered evaluation objects
EVAL_METRICS_CLASS_DICT = {
    config_fields.mock: MockEvaluation,
    config_fields.classification: ClassificationEvaluation,
    config_fields.ranking: RankingEvaluation,
}

class EvaluatorBuilder(object):
    """Evaluator Builder & Interface.

    Config:
        RankEvaluation
            All
        RankEvaluation
            Shape.Cup
            
    """
    def __init__(self, eval_config):
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

        self.eval_config = eval_config

        # Validation check of the eval_config
        minimum_requirements_of_config = [
            config_fields.container_size,
            config_fields.embedding_size,
            config_fields.logit_size,
            config_fields.evaluation,
            config_fields.database,
        ]
        for req in minimum_requirements_of_config:
            if not req in self.eval_config:
                raise ValueError('''The given configuration is not legal. It should
                    contain `evaluation`, `container_size`, `embedding_size` and `database`.''')
        self.evaluation_types = self.eval_config[config_fields.evaluation]
        if not self.evaluation_types:
            raise ValueError('No any evaluation is given in eval_config.')

        # TODO @kv: Consider seperating config_parser.

        # collect attribute types
        qeury_attribute_names = []
        for _eval_type in self.eval_config[config_fields.evaluation]:
            if _eval_type in EVAL_METRICS_CLASS_DICT:
                per_eval_qeury_commands = self.eval_config[config_fields.evaluation][_eval_type]
                if per_eval_qeury_commands:
                    for cmd in per_eval_qeury_commands:
                        qeury_attribute_names.append(cmd)
                else:
                    # None of the commands exist, what process we should do?
                    pass
        self.qeury_attribute_names = set(qeury_attribute_names)
        # TODO: Handle the special attributname, like AllAttributes, Overall

        # If no database is given, some evaluation can not be execute.
        # TODO @kv: How to build evaluator in such situation?
        if not self.eval_config[config_fields.database]:
            print ('WARNING: No database is assigned, some evaluations can not be executed.')
            # NOTE @kv: Should we ignore given attribure names & use category as attr_name?

        # allocate the shared embedding container
        embedding_size = self.eval_config[config_fields.embedding_size]
        container_size = self.eval_config[config_fields.container_size]
        logit_size = self.eval_config[config_fields.logit_size]
        self.embedding_container = EmbeddingContainer(embedding_size, logit_size, container_size)
        self.attribute_container = AttributeContainer() # NOTE: modify its implementation

        # Build evaluations
        self._build()

        # Allocate general query interface
        database_type = self.eval_config[config_fields.database]
        if not database_type:
            # TODO @kv: consistent check with query condition
            self.query_interface = None
        else:
            self.query_interface = QueryInterface(database_type)

    def _build(self):
        """
            Build evalution in list
        """

        try:
            self.evaluation_types
        except:
            raise AttributeError('Evaluation list is not given.')

        self.evaluations = {}
        self.evaluations_need_query = []
        for _eval_type in self.evaluation_types:
            per_eval_config = self.eval_config[config_fields.evaluation][_eval_type]
            _evaluation = EVAL_METRICS_CLASS_DICT[_eval_type](per_eval_config,
                                                              self.embedding_container,
                                                              self.attribute_container)
            self.evaluations[_eval_type] = _evaluation
            if hasattr(_evaluation, 'attribute_container'):
                self.evaluations_need_query.append(_eval_type)
            # TODO @kv: consistent check between attribute & configs.

    def add_image_id_and_embedding(self, image_id, label_id, embedding, logit=None):
        """Add embedding and label for a sample to be used for evaluation.
           If the query attribute names are given in config, this function will
           search them on database automatically.

        Args:
            image_id, integer:
                A integer identitfier for the image.
            label_id, integer:
                An index of label.
            embedding, list or numpy array:
                Embedding, feature vector
        """
        self.embedding_container.add(image_id, label_id, embedding, logit)
        # verbose for developing stage.
        if self.embedding_container.counts % 1000:
            print ('{} embeddings are added.'.format(self.embedding_container.counts))
        
        # TODO: consider move `add_image_id_and_query_attribute` here.
        # Collect all `attribute_name`
        # if not evaluations_need_query: no quries are needed.

        if self.query_interface:
            queried_attributes = self.query_interface.query(image_id, self.qeury_attribute_names)
            self.attribute_container.add(image_id, queried_attributes)

    
    def evaluate(self):
        """Execute given evaluations and returns a dictionary of metrics.
        
          Return:
            total_metrics, dict:

          1. 
            - img_id -> all attributes
        
          2. Create attribute containers with respect to each evlautions
                - query database with eval_config
        """

        total_metrics = {}

        for _eval_type, _evaluation in self.evaluations.items():
            metrics = _evaluation.compute()
            total_metrics[_eval_type] = metrics

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
        pass
        
    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        self.embedding_container.clear()
        self.attribute_container.clear()