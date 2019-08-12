import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


from abc import ABCMeta
from abc import abstractmethod


class DatabaseReaderBase(object):
    """Wrapper for reader
    """
    def __init__(self):
        pass

    @abstractmethod
    def query_attributes_by_instance_id(self):
        pass
