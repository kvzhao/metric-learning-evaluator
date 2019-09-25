"""
    FacenetEvaluation is the implementation referred to the repo:
        https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw
"""
from __future__ import division

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import math
from random import shuffle
import itertools
import numpy as np
from collections import defaultdict
from collections import namedtuple
from collections import Counter
from sklearn import metrics
from scipy import interpolate
from scipy.optimize import brentq

from metric_learning_evaluator.core.standard_fields import MetricStandardFields as metric_fields
from metric_learning_evaluator.core.standard_fields import AttributeStandardFields as attribute_fields
from metric_learning_evaluator.core.standard_fields import EvaluationStandardFields as eval_fields
from metric_learning_evaluator.core.standard_fields import FacenetEvaluationStandardFields as facenet_fields

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.data_tools.result_container import ResultContainer
from metric_learning_evaluator.evaluations.evaluation_base import MetricEvaluationBase

from metric_learning_evaluator.index.utils import euclidean_distance_filter
from metric_learning_evaluator.metrics.classification_metrics import ClassificationMetrics

from metric_learning_evaluator.utils.sample_strategy import SampleStrategy
from metric_learning_evaluator.core.standard_fields import SampleStrategyStandardFields as sample_fields


class FacenetEvaluation(MetricEvaluationBase):

    def __init__(self, config, mode=None):
        """
          FaceNet evaluates accuracy with pair instances.

          metric functions:
            - Top_k accuracy of same pairs
            - AUC
            - EER

          distance measures:

          Results
          -------------------------------------------------
            Accuracy: 0.99650+-0.00252
            Validation Rate: 0.98367+-0.00948 @ FAR=0.00100
            Area Under Curve (AUC): 1.000
            Equal Error Rate (EER): 0.004
          -------------------------------------------------
        """
        super(FacenetEvaluation, self).__init__(config, mode)

        print('Create {}'.format(self.evaluation_name))

        # Preprocess Configurations and check legal
        self._must_have_config = [
            eval_fields.distance_measure,
            eval_fields.sampling
        ]

        self._default_values = {
            eval_fields.distance_measure: {
                eval_fields.threshold: {
                    eval_fields.start: 0.01,
                    eval_fields.end: 0.7,
                    eval_fields.step: 0.01
                }
            },
            eval_fields.sampling: {
                facenet_fields.sample_ratio: 0.2,
                facenet_fields.class_sample_method: facenet_fields.random_sample
            }
        }
        # metrics with condition
        self._metric_with_threshold = [
            metric_fields.accuracy,
            metric_fields.validation_rate,
            metric_fields.false_accept_rate,
            metric_fields.true_positive_rate,
            metric_fields.false_positive_rate,
        ]
        # metrics without condition
        self._metric_without_threshold = [
            metric_fields.mean_accuracy,
            metric_fields.mean_validation_rate,
            metric_fields.area_under_curve,
        ]

        # Set default values for must-have configs
        for _config in self._must_have_config:
            if _config not in self.metrics:
                if _config in self._default_values:
                    pass
                else:
                    print('WARNING: {} should be assigned'.format(_config))
            else:
                print('Use assigned {}: {}'.format(_config, self.metrics[_config]))

        # Set distance thresholds by config
        distance_config = self.distance_measure
        distance_thres = distance_config[eval_fields.threshold]
        dist_start = distance_thres[eval_fields.start]
        dist_end = distance_thres[eval_fields.end]
        dist_step = distance_thres[eval_fields.step]
        # TODO @kv: Do we need sanity check for start < end?
        if dist_start > dist_end:
            raise ValueError('FaceEvaluation: distance threshold start > end')
        self._distance_thresholds = np.arange(dist_start, dist_end, dist_step)

        # Attributes
        if len(self.attributes) == 0:
            self._has_attribute = False
        elif len(self.attributes) == 1:
            if attribute_fields.all_classes in self.attributes:
                self._has_attribute = False
            elif attribute_fields.all_attributes in self.attributes:
                self._has_attribute = True
        else:
            self._has_attribute = True
        self.show_configs()

    @property
    def metric_names(self):
        _metric_names = []
        for _metric_name, _content in self.metrics.items():
            if not self.metrics.get(_metric_name, False):
                continue
            if _content is None:
                continue
            for _attr_name in self.attributes:
                if _metric_name in self._metric_without_threshold:
                    _name = '{}/{}'.format(_attr_name, _metric_name)
                    _metric_names.append(_name)
                if _metric_name in self._metric_with_threshold:
                    for threshold in self._distance_thresholds:
                        _name = '{}/{}@thres={}'.format(
                            _attr_name, _metric_name, threshold)
                        _metric_names.append(_name)
        return _metric_names

    def compute(self, embedding_container):
        """Procedure:
            - prepare the pair list for eval set
            - compute distance
            - calculate accuracy
        """
        self.result_container = ResultContainer()

        for group_cmd in self.group_commands:
            instance_ids = embedding_container.get_instance_id_by_group_command(group_cmd)
            if len(instance_ids) == 0:
                continue

            self._pair_measure(
                group_cmd, instance_ids, embedding_container)

        return self.result_container

    def _pair_measure(self,
                      attr_name,
                      instance_ids,
                      embedding_container):
        sampling_config = self.sampling
        label_ids = embedding_container.get_label_by_instance_ids(instance_ids)

        sampler = SampleStrategy(instance_ids, label_ids)
        sampled_pairs = sampler.sample_pairs(
            class_sample_method=sampling_config[sample_fields.class_sample_method],
            instance_sample_method=sampling_config[sample_fields.instance_sample_method],
            num_of_pairs=sampling_config[sample_fields.num_of_pairs],
            ratio_of_positive_pair=sampling_config[sample_fields.ratio_of_positive_pair],)

        # fetch instance ids and compute distances at once.
        pair_a_embeddings = embedding_container.get_embedding_by_instance_ids(
            sampled_pairs[sample_fields.pair_A])
        pair_b_embeddings = embedding_container.get_embedding_by_instance_ids(
            sampled_pairs[sample_fields.pair_B])
        ground_truth_is_same = np.asarray(sampled_pairs[sample_fields.is_same])

        # TODO: Change naming or use other functions
        predicted_is_same = euclidean_distance_filter(pair_a_embeddings,
                                                      pair_b_embeddings,
                                                      self._distance_thresholds)

        accuracy, tpr, fpr, val = [], [], [], []
        for threshold in self._distance_thresholds:
            classification_metrics = ClassificationMetrics()

            classification_metrics.add_inputs(
                predicted_is_same[threshold], ground_truth_is_same)
            if self.metrics.get(metric_fields.accuracy, True):
                self.result_container.add(
                    attr_name,
                    metric_fields.accuracy,
                    classification_metrics.accuracy,
                    condition={'thres': threshold})
            if self.metrics.get(metric_fields.validation_rate, True):
                self.result_container.add(
                    attr_name,
                    metric_fields.validation_rate,
                    classification_metrics.validation_rate,
                    condition={'thres': threshold})
            if self.metrics.get(metric_fields.false_accept_rate, True):
                self.result_container.add(
                    attr_name,
                    metric_fields.false_accept_rate,
                    classification_metrics.false_accept_rate,
                    condition={'thres': threshold})
            if self.metrics.get(metric_fields.true_positive_rate, True):
                self.result_container.add(
                    attr_name,
                    metric_fields.true_positive_rate,
                    classification_metrics.true_positive_rate,
                    condition={'thres': threshold})
            if self.metrics.get(metric_fields.false_positive_rate, True):
                self.result_container.add(
                    attr_name,
                    metric_fields.false_positive_rate,
                    classification_metrics.false_positive_rate,
                    condition={'thres': threshold})
            accuracy.append(classification_metrics.accuracy)
            tpr.append(classification_metrics.true_positive_rate)
            fpr.append(classification_metrics.false_positive_rate)
            val.append(classification_metrics.validation_rate)
            classification_metrics.clear()

        if self.metrics.get(metric_fields.mean_accuracy, True):
            self.result_container.add(
                attr_name,
                metric_fields.mean_accuracy,
                np.mean(accuracy))
        print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        if self.metrics.get(metric_fields.mean_validation_rate, True):
            self.result_container.add(
                attr_name,
                metric_fields.mean_validation_rate,
                np.mean(val))
        if self.metrics.get(metric_fields.area_under_curve, True):
            self.result_container.add(
                attr_name,
                metric_fields.area_under_curve,
                metrics.auc(fpr, tpr))
        # TODO: Problematic AUC value
        auc = metrics.auc(fpr, tpr)
        print('Area Under Curve (AUC): %1.3f' % auc)
        # TODO: Problematic EER value
        # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr, fill_value="extrapolate")(x), 0., 1.)
        # print('Equal Error Rate (EER): %1.3f' % eer)
