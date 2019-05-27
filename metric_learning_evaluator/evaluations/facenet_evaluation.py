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

from collections import defaultdict
from collections import namedtuple
from collections import Counter


# For verbose
from pprint import pprint


class FacenetEvaluationStandardFields(object):
    """Define fields used only in Facenet evaluation
        which may assign in `option` section in config.
    """

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
            if _config not in self.configs:
                if _config in self._default_values:
                    self.configs[_config] = self._default_values[_config]
                else:
                    print('WARNING: {} should be assigned'.format(_config))
            else:
                print('Use assigned {}: {}'.format(_config, self.configs[_config]))

        # Set distance thresholds by config
        distance_config = self.configs[eval_fields.distance_measure]
        distance_thres = distance_config[eval_fields.threshold]
        dist_start = distance_thres[eval_fields.start]
        dist_end = distance_thres[eval_fields.end]
        dist_step = distance_thres[eval_fields.step]
        # TODO @kv: Do we need sanity check for start < end?
        self.distance_thresholds = np.arange(dist_start, dist_end, dist_step)

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
                for threshold in self.distance_thresholds:
                    _name = '{}/{}@thres={}'.format(
                        _attr_name, _metric_name, threshold)
                    _metric_names.append(_name)
        return _metric_names


    def compute(self, embedding_container, attribute_container=None):
        """Procedure:
            - prepare the pair list for eval set
            - compute distance
            - calculate accuracy
        """
        result_container = ResultContainer()

        has_database = self.configs.has_database
        # sample pairs

        if not(attribute_container is None and has_database):
            # With attribute container & database
            print(attribute_container.attribute_names)
        else:
            # Without attribute container
            pass

    def old_compute(self, embedding_container, attribute_container=None):
        """Procedure:
            - prepare the pair list for eval set
            - compute distance
            - calculate accuracy

          NOTE: Attributes are NOT supported yet.
        """

        # numpy array
        img_ids = embedding_container.instance_ids

        # configs
        sampling_config = self.sampling
        num_of_pairs = sampling_config[facenet_fields.num_of_pairs]
        ratio_of_class = sampling_config[facenet_fields.ratio_of_class]
        num_of_instance_per_class = sampling_config[facenet_fields.num_of_instance_per_class]
        class_sample_method = sampling_config[facenet_fields.class_sample_method]

        assert len(img_ids) == embedding_container.embeddings.shape[0]

        result_container = ResultContainer()

        if not self._has_attribute:
            # NOTE: Assume attribute == `all_classes`
            attribute = attribute_fields.all_classes
            # @kv: load pair list from file or generate automatically
            pairs = self._generate_pairs(embedding_container,
                                         num_of_pairs,
                                         ratio_of_class,
                                         num_of_instance_per_class,
                                         class_sample_method)

            # fetch instance ids and compute distances at once.
            pair_a_embeddings = embedding_container.get_embedding_by_instance_ids(
                pairs[facenet_fields.pairA])
            pair_b_embeddings = embedding_container.get_embedding_by_instance_ids(
                pairs[facenet_fields.pairB])
            ground_truth_is_same = np.asarray(pairs[facenet_fields.is_same])

            # TODO @kv: choose distance function
            predicted_is_same = euclidean_distance_filter(pair_a_embeddings,
                                                          pair_b_embeddings,
                                                          self._distance_thresholds)

            for threshold in self._distance_thresholds:
                classification_metrics = ClassificationMetrics()
                classification_metrics.add_inputs(
                    predicted_is_same[threshold], ground_truth_is_same)
                result_container.add(
                    attribute,
                    metric_fields.accuracy,
                    classification_metrics.accuracy,
                    condition={'thres': threshold})
                result_container.add(
                    attribute,
                    metric_fields.validation_rate,
                    classification_metrics.validation_rate,
                    condition={'thres': threshold})
                result_container.add(
                    attribute,
                    metric_fields.false_accept_rate,
                    classification_metrics.false_accept_rate,
                    condition={'thres': threshold})
                result_container.add(
                    attribute,
                    metric_fields.true_positive_rate,
                    classification_metrics.true_positive_rate,
                    condition={'thres': threshold})
                result_container.add(
                    attribute,
                    metric_fields.false_positive_rate,
                    classification_metrics.false_positive_rate,
                    condition={'thres': threshold})
                classification_metrics.clear()

        else:
            # Has attributes
            pass

        # Fetch results
        print('{} compute done.'.format(self.evaluation_name))

        return result_container

    @staticmethod
    def _generate_pairs(embedding_container,
                        num_of_pairs,
                        ratio_of_class,
                        num_of_instance_per_class,
                        class_sample_method
                        ):
        """Image Pair & Sampling Strategy

          Args:

          Return:
            pairs, dict of list:

          A: {a1, a2, ..., aN} N instances in class A
          B: {b1, b2, ..., bM} M instances in class B
        """
        pairs = defaultdict(list)
        instance_ids = embedding_container.instance_ids
        label_ids = embedding_container.get_label_by_instance_ids(instance_ids)
        num_instance_ids = len(instance_ids)
        num_label_ids = len(label_ids)
        class_distribution = Counter(label_ids)
        classes = list(class_distribution.keys())
        num_classes = len(classes)

        assert num_label_ids == num_instance_ids

        # Randomly sample several data
        instance_ids = np.asarray(instance_ids)
        label_ids = np.asarray(label_ids)

        num_sampled_classes = math.ceil(ratio_of_class * num_classes)
        if ratio_of_class == 1.0:
            sampled_classes = classes
        else:
            if class_sample_method == facenet_fields.random_sample:
                sampled_classes = np.random.choice(classes, num_sampled_classes)
            elif class_sample_method == facenet_fields.amount_weighted:
                raise NotImplementedError
            elif class_sample_method == facenet_fields.amount_inverse_weighted:
                raise NotImplementedError
        num_sampled_classes = len(sampled_classes)

        num_of_pair_counter = 0
        all_combinations = list(itertools.combinations_with_replacement(sampled_classes, 2))
        shuffle(all_combinations)
        print('num of all comb. {}'.format(len(all_combinations)))
        print('num of sampled classes {}'.format(num_sampled_classes))

        # TODO @kv: Stack to record  seen class, for per_instance and has instance shown
        class_histogram = {}
        for _class in sampled_classes:
            class_histogram[_class] = 0

        sufficient_sample = False
        for class_a, class_b in all_combinations:
            num_img_class_a = class_distribution[class_a]
            num_img_class_b = class_distribution[class_b]
            num_sampled_img_per_class = min(num_img_class_a, num_img_class_b, num_of_instance_per_class)

            # statistics
            class_histogram[class_a] += num_sampled_img_per_class
            class_histogram[class_b] += num_sampled_img_per_class

            sampled_img_id_class_a = np.random.choice(
                embedding_container.get_instance_ids_by_label(class_a), num_sampled_img_per_class)
            sampled_img_id_class_b = np.random.choice(
                embedding_container.get_instance_ids_by_label(class_b), num_sampled_img_per_class)

            # Add instances in pair
            for img_class_a, img_class_b in zip(sampled_img_id_class_a, sampled_img_id_class_b):
                is_same = class_a == class_b
                pairs[facenet_fields.pairA].append(img_class_a)
                pairs[facenet_fields.pairB].append(img_class_b)
                pairs[facenet_fields.is_same].append(is_same)
                num_of_pair_counter += 1

            # check the strategic stop criteria
            if num_of_pair_counter > num_of_pairs:
                sufficient_sample = True
                """Find out which classes are less than requirements and sample them directly.
                """
                for _class, _counts in class_histogram.items():
                    total_num_of_instance = len(embedding_container.get_instance_ids_by_label(_class))

                    if _counts <= num_of_instance_per_class < total_num_of_instance:
                        sufficient_sample = False
                        break
            if sufficient_sample:
                break

        print('{} pairs are generated.'.format(num_of_pair_counter))
        return pairs

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
