
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

from abc import ABCMeta
from abc import abstractmethod


class FeatureExtractorBase(object):

    __metaclass__ = ABCMeta


    def __init__(self, pb_model_path, resize):

        self._pb_model_path = pb_model_path
        self._resize = resize

        self._model_status = None

    @abstractmethod
    def _model_init(self):
        pass

    @abstractmethod
    def extract(self, images):
        pass