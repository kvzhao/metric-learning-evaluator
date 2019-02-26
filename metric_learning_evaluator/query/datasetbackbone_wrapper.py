
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from metric_learning_evaluator.query.wrapper_base import DatabaseWrapperBase

from scutils.scdata import DatasetBackbone

class DatasetBackboneWrapper(object):

    def __init__(self):
        pass