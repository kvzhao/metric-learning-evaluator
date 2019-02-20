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
import random
import itertools
import numpy as np
from core.eval_standard_fields import MetricStandardFields as metric_fields
from core.eval_standard_fields import AttributeStandardFields as attribute_fields

from evaluator.data_container import EmbeddingContainer
from evaluator.data_container import AttributeContainer
from evaluator.data_container import ResultContainer
from evaluator.evaluation_base import MetricEvaluationBase

from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score

# from metrics.scores import calculate_positive_by_distance
from metrics.distances import euclidean_distance
from metrics.distances import euclidean_distance_filter

from metrics.classification_metrics import ClassificationMetrics

from collections import defaultdict
from collections import namedtuple
from collections import Counter
# For verbose
from pprint import pprint

class FacenetEvaluationStandardFields(object):
    """Define fields used only in Facenet evaluation
        which may assign in `option` section in config.
    """

    pairA = 'pairA'
    pairB = 'pairB'
    is_same = 'is_same'
    
    # old
    uniform_class = 'uniform_class'
    random_sample = 'random_sample'
    sample_method = 'sample_method'
    sample_ratio = 'sample_ratio'
    path_pairlist = 'path_pairlist'

    # new
    sample_method = 'sample_method'
    sample_ratio = 'sample_ratio'
    class_sample_method = 'class_sample_method'
    ratio_of_class = 'ratio_of_class'
    ratio_of_image_per_class = 'ratio_of_image_per_class'


facenet_fields = FacenetEvaluationStandardFields

class FacenetEvaluation(MetricEvaluationBase):

    def __init__(self, config):
        """
          FaceNet evaluates accuracy with pair images.

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
        super(FacenetEvaluation, self).__init__(config)
        
        print ('Create {}'.format(self._evaluation_name))

        # Preprocess Configurations and check legal
        self._available_metrics = [
            metric_fields.accuracy, 
        ]

        self._must_have_metrics = [
            metric_fields.distance_threshold,
            facenet_fields.sample_method,
            facenet_fields.sample_ratio
        ]

        # TODO: How to set default value?

        self._default_values = {
            metric_fields.distance_threshold: [0.5, 1.0, 1.5],
            facenet_fields.sample_method: facenet_fields.uniform_class,
            facenet_fields.sample_ratio: 0.2,
            facenet_fields.class_sample_method: facenet_fields.random_sample,
        }

        # Set default values for must-have metrics
        for _metric in self._must_have_metrics:
            if not _metric in self._eval_metrics:
                if _metric in self._default_values:
                    self._eval_metrics[_metric] = self._default_values[_metric]
                else:
                    print ("WARNING: {} should be assigned".format(_metric))
            else:
                print ('Use assigned {}: {}'.format(_metric, self._eval_metrics[_metric]))

        ## attributes
        if len(self._eval_attributes) == 0:
            self._has_attribute = False
        elif len(self._eval_attributes) == 1:
            if attribute_fields.all_classes in self._eval_attributes:
                self._has_attribute = False
            elif attribute_fields.all_attributes in self._eval_attributes:
                self._has_attribute = True
        else:
            self._has_attribute = True

        self.show_configs()

    def compute(self, embedding_container, attribute_container=None):
        """Procedure:
            - prepare the pair list for eval set
            - compute distance
            - calculate accuracy

          NOTE: Attributes are NOT supported yet.
        """

        # numpy array
        img_ids = embedding_container.image_ids

        # configs
        pair_sampling_config = self._eval_metrics[metric_fields.pair_sampling]
        distance_thresholds = self._eval_metrics[metric_fields.distance_threshold]

        ratio_of_class = pair_sampling_config[facenet_fields.ratio_of_class]
        ratio_of_image_per_class = pair_sampling_config[facenet_fields.ratio_of_image_per_class]
        class_sample_method = pair_sampling_config[facenet_fields.class_sample_method]

        assert len(img_ids) == embedding_container.embeddings.shape[0]

        result_container = ResultContainer(self._eval_metrics, self._eval_attributes)

        if not self._has_attribute:
            # attribute == all_classes
            attribute = attribute_fields.all_classes
            # @kv: load pair list from file or generate automatically
            pairs = self._generate_pairs(embedding_container,
                                         ratio_of_class,
                                         ratio_of_image_per_class,
                                         class_sample_method)

            # fetch image ids and compuate distances at once.
            pair_A_embeddings = embedding_container.get_embedding_by_image_ids(
                pairs[facenet_fields.pairA])
            pair_B_embeddings = embedding_container.get_embedding_by_image_ids(
                pairs[facenet_fields.pairB])
            groundtruth_is_same = np.asarray(pairs[facenet_fields.is_same])

            # TODO @kv: choose distance function
            predicted_is_same = euclidean_distance_filter(pair_A_embeddings,
                                                          pair_B_embeddings,
                                                          distance_thresholds)

            for threshold in distance_thresholds:
                classification_metrics = ClassificationMetrics()
                classification_metrics.add_inputs(predicted_is_same[threshold], groundtruth_is_same)
                result_container.add(attribute,
                    metric_fields.accuracy, threshold, classification_metrics.accuracy)
                result_container.add(attribute,
                    metric_fields.validation_rate, threshold, classification_metrics.validation_rate)
                result_container.add(attribute, 
                    metric_fields.false_accept_rate, threshold, classification_metrics.false_accept_rate)
                result_container.add(attribute,
                    metric_fields.true_positive_rate, threshold, classification_metrics.true_positive_rate)
                result_container.add(attribute,
                    metric_fields.false_positive_rate, threshold, classification_metrics.false_positive_rate)
                classification_metrics.clear()
        
        else:
            # Has attributes
            pass

        # Fetch results
        print ("{} compute done.".format(self.evaluation_name))

        return result_container.results


    def _generate_pairs(self,
                        embedding_container,
                        ratio_of_class,
                        ratio_of_image_per_class,
                        class_sample_method
                        ):
        """Image Pair & Sampling Strategy

          Args:

          Return:
            pairs, dict of list:

          A: {a1, a2, ..., aN} N images in class A
          B: {b1, b2, ..., bM} M images in class B
        """
        pairs = defaultdict(list)
        image_ids = embedding_container.image_ids
        label_ids = embedding_container.get_label_by_image_ids(image_ids)
        num_image_ids = len(image_ids)
        num_label_ids = len(label_ids)
        class_histogram = Counter(label_ids)
        classes = list(class_histogram.keys())
        num_classes = len(classes)

        assert num_label_ids == num_image_ids

        num_sampled_classes = int(num_classes * ratio_of_class)

        # Randomly sample several data
        image_ids = np.asarray(image_ids)
        label_ids = np.asarray(label_ids)

        if class_sample_method == facenet_fields.random_sample:
            sampled_classes = np.random.choice(classes, num_sampled_classes)

        num_pairs = 0
        for class_a, class_b in list(itertools.combinations_with_replacement(sampled_classes, 2)):
            num_img_class_a = math.ceil(class_histogram[class_a] * ratio_of_image_per_class)
            num_img_class_b = math.ceil(class_histogram[class_b] * ratio_of_image_per_class)
            num_sampled_img_per_class = min(num_img_class_a, num_img_class_b)

            sampled_img_id_class_a = np.random.choice(
                embedding_container.get_image_ids_by_label(class_a), num_sampled_img_per_class)
            sampled_img_id_class_b = np.random.choice(
                embedding_container.get_image_ids_by_label(class_b), num_sampled_img_per_class)
            for img_class_a, img_class_b in zip(sampled_img_id_class_a, sampled_img_id_class_b):
                is_same = class_a == class_b
                pairs[facenet_fields.pairA].append(img_class_a)
                pairs[facenet_fields.pairB].append(img_class_b)
                pairs[facenet_fields.is_same].append(is_same)
                num_pairs += 1

        print ('{} pairs are generated.'.format(num_pairs))
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