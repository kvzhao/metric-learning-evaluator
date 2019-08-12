"""
    Mask Detector Base Class.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

from abc import ABCMeta
from abc import abstractmethod

class MaskDetectorStandardFields:
    num_detections = 'num_detections'
    detection_classes = 'detection_classes'
    detection_boxes = 'detection_boxes'
    detection_scores = 'detection_scores'
    detection_masks = 'detection_masks'


class MaskDetectorBase(object):

    def __init__(self, pb_model_path, labelmap_path, num_classes,
                       image_height, image_width):
        
        self._pb_model_path = pb_model_path
        self._labelmap_path = labelmap_path
        self._num_classes = num_classes
        self._image_height = image_height
        self._image_widhth = image_width

        self._model_status = None # use this?
        self._labelmap = None

    @abstractmethod
    def _model_init(self):
        pass

    @abstractmethod
    def _run_inference(self, images):
        pass

    def detect(self, images):
        pass
