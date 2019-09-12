"""Image Retrieval
     - Detector
     - Feature Extractor
     - Index Agent
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np


from metric_learning_evaluator.inference.components.detector import Detector
from metric_learning_evaluator.inference.components.extractor import FeatureExtractor

from metric_learning_evaluator.data_tools.image_object import ImageObject
from metric_learning_evaluator.data_tools.feature_object import FeatureObject
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer

from metric_learning_evaluator.inference.utils.image_utils import bbox_ratio_to_xywh
from metric_learning_evaluator.inference.utils.image_utils import crop_and_resize
from metric_learning_evaluator.inference.utils.read_json_labelmap import read_json_labelmap

from metric_learning_evaluator.index.utils import indexing_array
from metric_learning_evaluator.index.utils import euclidean_distance
# use index agent instead
from metric_learning_evaluator.index.agent import IndexAgent

from metric_learning_evaluator.core.standard_fields import DetectorStandardFields


class ImageRetrievalStandardFields:
    detector_settings = 'detector_settings'
    extractor_settings = 'extractor_settings'
    num_classes = 'num_classes'
    model_path = 'model_path'
    labelmap_path = 'labelmap_path'
    image_size = 'image_size'
    database_path = 'database_path'

config_fields = ImageRetrievalStandardFields
detector_fields = DetectorStandardFields


class ImageRetrieval(object):
    """Retrieval
        - Input: an single image with 3d channels
        - Output: an ImageObject describes the image and instances.
      TODO: Change the initialization procedure
    """
    def __init__(self, configs):
        self._configs = configs

        self._image_counts = 0
        self._instance_counts = 0
        self._build()
        print('Retrieval Module Initialized.')

    def _build(self):
        """
          Build model from given paths
            the information is passed by config dict
        """
        detector_settings = self._configs[config_fields.detector_settings]
        extractor_settings = self._configs[config_fields.extractor_settings]

        detector_num_classes = detector_settings[config_fields.num_classes]
        detector_model_path = detector_settings[config_fields.model_path]
        detector_labelmap_path = detector_settings[config_fields.labelmap_path]
        extractor_model_path = extractor_settings[config_fields.model_path]
        extractor_image_size = extractor_settings[config_fields.image_size]
        self._extractor_image_size = extractor_image_size

        # TODO: Choose model type?
        self._detector = Detector(
            pb_model_path=detector_model_path,
            labelmap_path=detector_labelmap_path,
            num_classes=detector_num_classes)

        self._extractor = FeatureExtractor(
            pb_model_path=extractor_model_path,
            resize=extractor_image_size)

        self._search_database_path = extractor_settings[
            config_fields.database_path]
        if self._search_database_path and not os.path.exists(self._search_database_path):
            self._search_database_path = None

        """Label Map"""
        label_map_path = self._configs[config_fields.labelmap_path]
        if label_map_path is not None:
            self._labelmap = read_json_labelmap(label_map_path)
        else:
            self._labelmap = {}

        self._agent = None
        self._container = None
        if self._search_database_path is not None:
            self._container = EmbeddingContainer(name='anchor_database')
            self._container.load(self._search_database_path)
            self._agent = IndexAgent(agent_type='HNSW',
                                     instance_ids=self._container.instance_ids,
                                     embeddings=self._container.embeddings)

    def _search(self, feature, top_k=10):
        _ids, _ = self._agent.search(feature, top_k)
        _ids = np.squeeze(_ids)
        retrieved_label_ids = self._container.get_label_by_instance_ids(_ids)
        retrieved_label_names = self._container.get_label_name_by_instance_ids(_ids)
        return retrieved_label_names, retrieved_label_ids

    def inference(self, image, image_id=None):
        """
          Args:
            image: Single image with 3D numpy array.
            image_object
          Return:
            image_object: ImageObject which contains full information of predicted results.
        """

        if image_id:
            _image_id = image_id
        else:
            _image_id = self._image_counts

        image_object = ImageObject(_image_id, image)
        img_height, img_width, _ = image.shape

        detected_result = self._detector.detect(image)
        det_bboxes = detected_result[detector_fields.detection_boxes]
        det_scores = detected_result[detector_fields.detection_scores]

        for inst_id, (bbox, score) in enumerate(zip(det_bboxes, det_scores)):
            formal_bbox = bbox_ratio_to_xywh(bbox, img_height, img_width)
            # TODO @kv: Reranking Strategy may be here.
            inst_img = crop_and_resize(image, formal_bbox, self._extractor_image_size)
            inst_feat = self._extractor.extract_feature(inst_img)
            image_object.add_instance(self._instance_counts, formal_bbox, feature=inst_feat)

            if not (self._container is None or self._agent is None):
                searched_instance_names, searched_instance_labels = self._search(inst_feat)

                image_object.update_instance(self._instance_counts,
                                             instance_label_id=searched_instance_labels[0],
                                             instance_label_name=searched_instance_names[0])
            else:
                image_object.update_instance(self._instance_counts,
                                             instance_label_id=0,
                                             instance_label_name='unknown_instance')
            self._instance_counts += 1
        self._image_counts += 1

        return image_object
