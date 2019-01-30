"""
    FacenetEvaluation is the implementation referred to the repo:
        https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw
"""
from __future__ import division

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from evaluator.data_container import EmbeddingContainer
from evaluator.data_container import AttributeContainer
from evaluator.data_container import ResultContainer
from evaluator.evaluation_base import MetricEvaluationBase

import numpy as np

from core.eval_standard_fields import MetricStandardFields as metric_fields

from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score

from metrics.distances import batch_euclidean_distances
from metrics.scores import calculate_positive_by_distance

from collections import namedtuple
# For verbose
from pprint import pprint

# TODO: Change name to PairEvaluation
Pair = namedtuple('Pair', 'img_A, img_B, issame')

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
            Validation rate: 0.98367+-0.00948 @ FAR=0.00100
            Area Under Curve (AUC): 1.000
            Equal Error Rate (EER): 0.004
        """
        super(FacenetEvaluation, self).__init__(config)
        
        print ('Create {}'.format(self._evaluation_name))

        # Preprocess Configurations
        self._available_metrics = [
            metric_fields.top_k,
        ]

        if self._eval_attributes:
            pass
        else:
            pass

        if self._eval_attributes:
            pass
        else:
            self._has_attribute = False

        self.show_configs()

        # Verbose
        print (self._config.get_per_eval_config(self.evaluation_name))


    def compute(self, embedding_container, attribute_container=None):
        """Procedure:
            - prepare the pair list for eval set
            - compute distance
            - calculate accuracy

        """

        img_ids = embedding_container.image_ids
        label_ids = embedding_container.get_label_by_image_ids(img_ids)
        embeddings = embedding_container.embeddings
        index_map = embedding_container.index_by_image_ids

        print ('shape of embeddings: {}'.format(embeddings.shape))
        print ('image ids: {}'.format(img_ids))
        print ('label ids: {}'.format(label_ids))

        assert(len(img_ids) == embeddings.shape[0])

        pairs = self._generate_pairs(img_ids, label_ids)

        # generate the img_id pair list
        for pair in pairs:
            print (pair)
        #pairs, distances, actual_issame = self._compute_pair_distances(img_ids, label_ids,
        #                                                               embeddings, index_map)
        # assert (len(pairs) == len(actual_issame) == len(distances))
        #actual_issame = np.array(actual_issame)
        #tpr, fpr, acc = calculate_positive_by_distance(distances, .2, actual_issame)
        #print (tpr, fpr, acc)

        #for pair, dist, issame in zip(pairs, distances, actual_issame):
        #    print (pair, dist, issame)

    def _generate_pairs(self, image_ids, label_ids, sample_method="all"):
        """Image Pair & Sampling Strategy

          Args:

            pairs: list of namedtuple
                Pair = (img_A, img_B, issame) 
                e.g.
                    A, B
                    img_1, img_2
                    img_1, img_3
                    img_1, img_4
                    ...
                    img_3, img_4,
                    img_4, img_5
            issame: list of boolean 
                It denotes corresponding pair is same class or not
            sample_method:
        
          Return:
            pairs, list of namedtuple:
                It consists with image_ids
        """
        pairs = []
        if sample_method is None or sample_method == 'all':
            for imgid_src, lid_src in zip(image_ids, label_ids):
                for imgid_tar, lid_tar in zip(image_ids, label_ids):
                    issame = lid_src == lid_tar
                    pairs.append(Pair(imgid_src, imgid_tar, issame))
        return pairs

    def _compute_pair_distances(self, image_ids, label_ids, embeddings, index_map):
        """Pair Distances:
            Generate image pair: (img_1, img_2, actual_issame) and set
            distance(embeddings[img_1], embeddings[img_2]) <= threshold to predicted_issame.

            # NOTE: How to change distance measure?

            Args:
              image_ids, list
              label_ids, list
              embeddings, 2D array
              index_map, dict: image_id to index of numpy array

            Returns:
              pairs, list of tuple
              distances, 1D numpy array
              actual_issame, list of boolean


        """

        pairs = []
        actual_issame = []
        distances = []
        for idx, (imgid_src, lid_src) in enumerate(zip(image_ids, label_ids)):
            for imgid_tar, lid_tar in zip(image_ids[idx:], label_ids[idx:]):
                pairs.append((imgid_src, imgid_tar))
                actual_issame.append(lid_src == lid_tar)
            src_embed = embeddings[index_map[imgid_src]]
            dist = batch_euclidean_distances(src_embed, embeddings)
            distances.extend(dist)
        distances = np.asarray(distances)
        return pairs, distances, actual_issame


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