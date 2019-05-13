import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


from abc import ABCMeta
from abc import abstractmethod

class DatabaseWrapperBase(object):
    """
      Wrapper should be re-defined
    """

    def __init__(self):
        pass

    @abstractmethod
    def query_image_ids_by_attribute(self):
        pass