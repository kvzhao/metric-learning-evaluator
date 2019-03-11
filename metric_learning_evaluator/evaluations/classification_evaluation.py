"""
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np


from metric_learning_evaluator.core.eval_standard_fields import AttributeStandardFields as attr_fields
from metric_learning_evaluator.core.eval_standard_fields import EvaluationStandardFields as eval_fields
from metric_learning_evaluator.core.eval_standard_fields import MetricStandardFields as metric_fields

from metric_learning_evaluator.data_tools.result_container import ResultContainer
from metric_learning_evaluator.evaluations.evaluation_base import MetricEvaluationBase

from metric_learning_evaluator.metrics.classification_metrics import ClassificationMetrics



class ClassificationEvaluation(MetricEvaluationBase):

    def __init__(self, config):
        super(ClassificationEvaluation, self).__init__(config)
        """Classification Evaluation
            The evaluation computes accuracy from given logits and return top_k accuracy.
            If attributes are provided, calculate top_k accuracy per attribute.
        """

        print ('Create {}'.format(self._evaluation_name))

        # Define the available metrics.
        self._available_metrics = [
            metric_fields.top_k,
        ]

    def compute(self, embedding_container, attribute_container=None):
        """Compute Accuracy.

            Get compute probabilities from logits and compare with label
            from embedding_container.
            
          Args:
            embedding_container, EmbeddingContainer 

            attribute_container, AttributeContainer

          Return:
            results, ResultContainer

        """

        if not isinstance(embedding_container.logits,
                         (np.ndarray, np.generic)):
            raise AttributeError('Logits should be provided when {} is performed'.format(self.evaluation_name))
        
        classification_metrics = ClassificationMetrics()

        img_ids = embedding_container.image_ids

        # NOTE: Make sure that length of inputs are equal.
        per_eval_attributes = self._configs.get_attributes(self.evaluation_name)
        per_eval_metrics = self._configs.get_metrics(self.evaluation_name)

        result_container = ResultContainer(per_eval_metrics, per_eval_attributes)

        if attr_fields.all_classes not in per_eval_attributes and attribute_container:
            has_attributes = True
        else:
            has_attributes = False

        # Evaluate overall if attribute not given
        if not has_attributes:
            pass
        else:
            # With attributes
            pass
 
        # Add parsed array and get results

        # Push them into result container

        return result_container.results