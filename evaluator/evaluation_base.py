"""
    The base object and data container for the metric learning evaluator.

    Brief intro:

        EmbeddingContainer: 
            Efficient object which handles the shared (globally) embedding vectors.

        AttributeContainer:
            Data object for maintaining attribute table locally in each EvaluationObejct.

        MetricEvaluationBase:

    @bird, dennis, kv
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from abc import ABCMeta
from abc import abstractmethod
import collections
from collections import namedtuple
from collections import OrderedDict
from collections import defaultdict

import logging
import numpy as np

class MetricEvaluationBase(object):    
    """Interface for Evaluation Object which serves as the functional building block.

        Usage:

    """
    __metaclass__ = ABCMeta

    def __init__(self, per_eval_config):
        """Base Object for Evaluation.
          <Customized>Evaluation is the functional object which 
                      executes computation with metric functions.

          Args:
            per_eval_config, dict:
                Configuration used for the EvaluationObject.
                TODO @kv: Define the format:

          Init function:
            parse the per_eval_config.

        """

        if per_eval_config and not isinstance(per_eval_config, dict):
            raise ValueError('Evaluation Config is a dictionary of required attributes.')

        self._per_eval_config = per_eval_config

        # TODO: Iterator for getting embeddings from given attribute_names
        self._evaluation_name = self.__class__.__name__

    @property
    def evaluation_name(self):
        return self._evaluation_name

    @abstractmethod
    def compute(self, embedding_container, attribute_container=None):
        """Compute metrics.
          Args:
            embedding_container, EmbeddingContainer:
                The embedding container is necessary.

            attribute_container, AttributeContrainer:
                The attribute container is optional, it can be `None` for some evaluation.

          Return:
            metrics, dict:
                TODO @kv: Define the standard return format.
        """

        # check the instance type.
        if not isinstance(embedding_container, EmbeddingContainer):
            raise ValueError('Embedded Conatiner is Needed.')
        if attribute_container and not isinstance(embedding_container, EmbeddingContainer):
            raise ValueError('Attribute Conatiner is Needed.')

        ## Do the customized computation.
