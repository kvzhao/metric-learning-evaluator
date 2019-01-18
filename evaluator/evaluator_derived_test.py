
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from abc import ABCMeta
from abc import abstractmethod
import collections
from collections import namedtuple
import logging
import numpy as np


from evaluator.evaluation_base import MetricEvaluationBase

class MetricEvaluatorDerived(MetricEvaluatorBase):
    pass