"""
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer

from metric_learning_evaluator.data_tools.result_container import ResultContainer
from metric_learning_evaluator.evaluations.evaluation_base import MetricEvaluationBase

from metric_learning_evaluator.metrics.classification_metrics import ClassificationMetrics
from metric_learning_evaluator.metrics.ranking_metrics import RankingMetrics

from metric_learning_evaluator.utils.sample_strategy import SampleStrategy

from metric_learning_evaluator.core.standard_fields import MetricStandardFields as metric_fields
from metric_learning_evaluator.core.standard_fields import AttributeStandardFields as attr_fields
from metric_learning_evaluator.core.standard_fields import SampleStrategyStandardFields as sample_fields
from metric_learning_evaluator.core.standard_fields import ClassificationEvaluationStandardFields as cls_fields


class ClassificationEvaluation(MetricEvaluationBase):

    def __init__(self, config, mode=None):
        super(ClassificationEvaluation, self).__init__(config, mode)
        """Classification Evaluation
            The evaluation computes accuracy from given logits and return top_k accuracy.
            If attributes are provided, calculate top_k accuracy per attribute.
        """

        # metrics with condition
        self._metric_with_threshold = [
            metric_fields.top_k_hit_accuracy,
        ]
        # metrics without condition
        self._metric_without_threshold = [
            metric_fields.mAP,
        ]

        print('Create {}'.format(self._evaluation_name))
        self.show_configs()

    @property
    def metric_names(self):
        _metric_names = []
        for _metric_name, _content in self.metrics.items():
            for _attr_name in self.attributes:
                if _content is None:
                    continue
                if _metric_name in self._metric_without_threshold:
                    _name = '{}/{}'.format(_attr_name, _metric_name)
                    _metric_names.append(_name)
                if _metric_name in self._metric_with_threshold:
                    # special condition
                    if _metric_name == metric_fields.top_k_hit_accuracy:
                        top_k_list = self.metrics[metric_fields.top_k_hit_accuracy]
                        for top_k in top_k_list:
                            _name = '{}/{}@k={}'.format(_attr_name, _metric_name, top_k)
                            _metric_names.append(_name)
        return _metric_names

    def compute(self, embedding_container):
        """Compute Accuracy.
            Get compute classification metrics with categorical scores and label from embedding_container.
          Args:
            embedding_container: EmbeddingContainer
          Return:
            results: ResultContainer
        """

        # check the size is non-zero
        if not isinstance(embedding_container.probabilities,
                          (np.ndarray, np.generic)):
            raise AttributeError('Logits should be provided when {} is performed'.format(self.evaluation_name))

        self.result_container = ResultContainer()

        # has attribute
        for group_cmd in self.group_commands:
            instance_ids_given_attribute = embedding_container.get_instance_id_by_group_command(group_cmd)
            if len(instance_ids_given_attribute) == 0:
                continue

            self._classification_measure(
                group_cmd, instance_ids_given_attribute, embedding_container)

        return self.result_container

    def _classification_measure(self,
                                attr_name,
                                instance_ids,
                                embedding_container):
        """
          Args:
            instance_ids:
            label_ids:
            probabilities:
            NOTE: Make sure instance_ids, label_ids and probabilities are in same order.
            TODO: Push the sampler here
        """
        cls_config = self.metrics
        sampling_config = self.sampling

        # all instance_ids & label_ids
        label_ids = embedding_container.get_label_by_instance_ids(instance_ids)
        sampler = SampleStrategy(instance_ids, label_ids)

        class_sample_method = sampling_config[sample_fields.class_sample_method]
        instance_sample_method = sampling_config[sample_fields.instance_sample_method]
        num_of_sampled_class = sampling_config[sample_fields.num_of_sampled_class]
        num_of_sampled_instance = sampling_config[sample_fields.num_of_sampled_instance_per_class]

        sampled = sampler.sample(
            class_sample_method,
            instance_sample_method,
            num_of_sampled_class,
            num_of_sampled_instance,
        )

        sampled_instance_ids = sampled[sample_fields.sampled_instance_ids]
        sampled_label_ids = sampled[sample_fields.sampled_label_ids]
        sampled_probabilities = embedding_container.get_probability_by_instance_ids(sampled_instance_ids)

        top_k_list = cls_config[metric_fields.top_k_hit_accuracy]

        for top_k in top_k_list:
            if top_k == 1:
                continue
            cls_metrics = RankingMetrics(top_k)
            hit_arrays = np.empty((sampled_probabilities.shape[0], top_k), dtype=np.bool)

            for _idx, gt_label_id in enumerate(sampled_label_ids):
                prob = sampled_probabilities[_idx]
                top_k_hit_ids = np.argsort(-prob)[:top_k]
                top_k_hit_ids = top_k_hit_ids == gt_label_id
                hit_arrays[_idx, ...] = top_k_hit_ids
            cls_metrics.add_inputs(hit_arrays)
            self.result_container.add(attr_name, cls_fields.top_k_hit_accuracy,
                                      cls_metrics.topk_hit_accuracy, condition={'k': top_k})
        self.result_container.add(attr_name, cls_fields.top_k_hit_accuracy,
                                  cls_metrics.top1_hit_accuracy, condition={'k': 1})
