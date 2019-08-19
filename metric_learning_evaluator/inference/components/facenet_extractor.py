# FeatureExtractor
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import numpy as np
import tensorflow as tf

from metric_learning_evaluator.inference.components.extractor_base import FeatureExtractorBase


class FeatureExtractor(FeatureExtractorBase):
    def __init__(self,
                 pb_model_path,
                 resize):
        super(FeatureExtractor, self).__init__(pb_model_path, resize)

        self._model_init()

    def _model_init(self):
        """Initialization"""
        # load labelmap
        # load graph
        self._graph = tf.Graph()
        with self._graph.as_default() as graph:
            graph_def = tf.GraphDef()
            
            with tf.gfile.GFile(self._pb_model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(graph_def, name='')
                
            self.sess = tf.Session(graph=graph)

            # get tensor
            try:
                self.images_placeholder =\
                    graph.get_tensor_by_name("input:0")
                self.embeddings =\
                    graph.get_tensor_by_name("embeddings:0")
                self._phase_train_placeholder =\
                    graph.get_tensor_by_name("phase_train:0")
            except Exception as e:
                print(e)
                print('Can not find tensor name starts with "<name>",'
                          'try "import/<name>"')
                self.images_placeholder = graph.get_tensor_by_name("import/input:0")
                self.embeddings = graph.get_tensor_by_name("import/embeddings:0")
                self._phase_train_placeholder = graph.get_tensor_by_name("import/phase_train:0")

        print('Feature Extractor Initialized, Model Loaded from : {}'.format(self._pb_model_path))

    def extract_feature(self, instance_images):
        """
        Args:
            images: ndarray, images of format (N, H, W, C). The given images are resized.
        Returns
            embedding: list of feature vector
        """

        if len(instance_images.shape) == 3:
            instance_images = np.expand_dims(instance_images, 0)
        feed_dict = {
            self.images_placeholder: instance_images,
            self._phase_train_placeholder: False
        }

        embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)

        return embeddings
