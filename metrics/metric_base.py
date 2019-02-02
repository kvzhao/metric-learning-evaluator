from abc import ABCMeta
from abc import abstractmethod


# TODO @jeff: add doc strings
class MetricBase(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    @property
    def is_empty(self):
        pass

    @abstractmethod
    def clear(self):
        pass
