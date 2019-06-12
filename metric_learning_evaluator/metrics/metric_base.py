
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from abc import ABCMeta
from abc import abstractmethod


# TODO @jeff: add doc strings
class MetricBase(object):

    __metaclass__ = ABCMeta

    # @property
    @abstractmethod
    def is_empty(self):
        pass

    @abstractmethod
    def clear(self):
        pass
