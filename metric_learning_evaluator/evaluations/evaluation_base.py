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

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.data_tools.attribute_container import AttributeContainer

import logging
import numpy as np

class MetricEvaluationBase(object):    
    """Interface for Evaluation Object which serves as the functional building block.

        Usage:

    """
    __metaclass__ = ABCMeta

    def __init__(self, config_parser):
        """Base Object for Evaluation.
          <Customized>Evaluation is the functional object which 
                      executes computation with metric functions.

          Args:
            config_parse, ConfigParse:
                Configuration used for the EvaluationObject.

          Init function:
            parse the per_eval_config.

        """

        #if config_parser and not isinstance(config_parser, ConfigParser):
        #    raise ValueError('Evaluation requires the ConfigParser object.')

        self._config_parser = config_parser

        # TODO: Iterator for getting embeddings from given attribute_names
        self._evaluation_name = self.__class__.__name__

        self._configs = self._config_parser.get_per_eval_config(self.evaluation_name)
        # preprocessing eval config in each customized evaluation
        self._metrics = self._config_parser.get_metrics(self.evaluation_name)
        self._attributes = self._config_parser.get_attributes(self.evaluation_name)

        # Verbose
        print (self._config_parser.get_per_eval_config(self.evaluation_name))

    @property
    def evaluation_name(self):
        return self._evaluation_name

    @abstractmethod
    def metric_names(self):
        pass

    def show_configs(self):
        print ('{} - Compute {} metrics over attributes: {}'.format(
            self.evaluation_name, ', '.join(self._metrics.keys()), ', '.join(self._attributes)))

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
            raise ValueError('Embedded Container is Needed.')
        if attribute_container and not isinstance(embedding_container, EmbeddingContainer):
            raise ValueError('Attribute Container is Needed.')

        ## Do the customized computation.

    @abstractmethod
    def save(self, path):
        pass