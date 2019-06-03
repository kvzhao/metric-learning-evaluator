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
from metric_learning_evaluator.metrics.standard_fields import MetricStandardFields as metric_fields
from metric_learning_evaluator.query.standard_fields import AttributeStandardFields as attribute_fields
from metric_learning_evaluator.evaluations.standard_fields import EvaluationStandardFields as eval_fields

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.data_tools.attribute_container import AttributeContainer
from metric_learning_evaluator.data_tools.result_container import ResultContainer
from metric_learning_evaluator.evaluations.evaluation_base import MetricEvaluationBase

from metric_learning_evaluator.index.utils import euclidean_distance_filter
from metric_learning_evaluator.metrics.classification_metrics import ClassificationMetrics

from metric_learning_evaluator.utils.sample_strategy import SampleStrategy
from metric_learning_evaluator.utils.sample_strategy import SampleStrategyStandardFields as sample_fields

from collections import defaultdict
from collections import namedtuple
from collections import Counter


# For verbose
from pprint import pprint


class FacenetEvaluationStandardFields(object):
    """Define fields used only in Facenet evaluation
        which may assign in `option` section in config.
    """

    ### NOTE @kv: Move these sampling option into sample_strategy

    # pair dict
    pairA = 'pairA'
    pairB = 'pairB'
    is_same = 'is_same'
    path_pairlist = 'path_pairlist'
    num_maximum_pairs = 'num_maximum_pairs'
    num_of_pairs = 'num_of_pairs'

    # used for distance threshold
    start = 'start'
    end = 'end'
    step = 'step'

    # sampling options
    sample_method = 'sample_method'
    sample_ratio = 'sample_ratio'
    ratio_of_class = 'ratio_of_class'
    ratio_of_instance_per_class = 'ratio_of_instance_per_class'
    num_of_instance_per_class = 'num_of_instance_per_class'

    # sampling methods
    class_sample_method = 'class_sample_method'
    random_sample = 'random_sample'
    amount_weighted = 'amount_weighted'
    amount_inverse_weighted = 'amount_inverse_weighted'


facenet_fields = FacenetEvaluationStandardFields


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
                    eval_fields.start: 0.5,
                    eval_fields.end: 1.5,
                    eval_fields.step: 0.2
                }
            },
            eval_fields.sampling: {
                facenet_fields.sample_ratio: 0.2,
                facenet_fields.class_sample_method: facenet_fields.random_sample
            }
        }

        # Set default values for must-have configs
        for _config in self._must_have_config:
            if _config not in self.metrics:
                if _config in self._default_values:
                    pass
                    #self.metrics[_config] = self._default_values[_config]
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
            if _content is None:
                continue
            for _attr_name in self.attributes:
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
        embedding_container,
        ):

        sampling_config = self.sampling

        label_ids = embedding_container.get_label_by_instance_ids(instance_ids)

        sampler = SampleStrategy(instance_ids, label_ids)
        sampled_pairs = sampler.sample_pairs(
            class_sample_method=sampling_config[sample_fields.class_sample_method],
            instance_sample_method=sampling_config[sample_fields.instance_sample_method],
            num_of_pairs=sampling_config[sample_fields.num_of_pairs],)

        # fetch instance ids and compute distances at once.
        pair_a_embeddings = embedding_container.get_embedding_by_instance_ids(
            sampled_pairs[sample_fields.pair_A])
        pair_b_embeddings = embedding_container.get_embedding_by_instance_ids(
            sampled_pairs[sample_fields.pair_B])
        ground_truth_is_same = np.asarray(sampled_pairs[sample_fields.is_same])

        predicted_is_same = euclidean_distance_filter(pair_a_embeddings,
                                                     pair_b_embeddings,
                                                     self._distance_thresholds)

        for threshold in self._distance_thresholds:
            classification_metrics = ClassificationMetrics()
            classification_metrics.add_inputs(
                predicted_is_same[threshold], ground_truth_is_same)
            self.result_container.add(
                attr_name,
                metric_fields.accuracy,
                classification_metrics.accuracy,
                condition={'thres': threshold})
            self.result_container.add(
                attr_name,
                metric_fields.validation_rate,
                classification_metrics.validation_rate,
                condition={'thres': threshold})
            self.result_container.add(
                attr_name,
                metric_fields.false_accept_rate,
                classification_metrics.false_accept_rate,
                condition={'thres': threshold})
            self.result_container.add(
                attr_name,
                metric_fields.true_positive_rate,
                classification_metrics.true_positive_rate,
                condition={'thres': threshold})
            self.result_container.add(
                attr_name,
                metric_fields.false_positive_rate,
                classification_metrics.false_positive_rate,
                condition={'thres': threshold})
            classification_metrics.clear()

    def _compute_roc(self):
        """ROC Curve
        """
        pass

    def _compute_err_rate(self):
        """Equal Error Rate (EER)
            Equal Error Rate (EER) is the point on the ROC curve
            that corresponds to have an equal probability of miss-classifying a positive or negative sample.
            This point is obtained by intersecting the ROC curve with a diagonal of the unit square.
        """
        pass
