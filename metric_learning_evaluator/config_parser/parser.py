import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import yaml
import inspect

# refactor
from metric_learning_evaluator.config_parser.standard_fields import ConfigStandardFields as config_fields
from metric_learning_evaluator.evaluations.standard_fields import EvaluationStandardFields as eval_fields
from metric_learning_evaluator.query.standard_fields import AttributeStandardFields as attr_fields
#from metric_learning_evaluator.metrics.standard_fields import MetricStandardFields as metric_fields
from metric_learning_evaluator.core.registered import REGISTERED_EVALUATION_OBJECTS

from pprint import pprint

class ConfigParser(object):
    """
      User Scenario:
        1. load from given config path
        2. get useful values to set containers
        3. get hyper-parameters for evaluation functions
        
        NOTE:
          evaluation_type: classification vs. evaluation_name: ClassificationEvaluation
    """

    def __init__(self, config_dict):
        """
          Args:
            configs: loaded dict from eval_config.yaml
        """

        self._configs = config_dict

        minimum_requirements_of_config = [
            config_fields.container_size,
            config_fields.evaluation_options,
            config_fields.database,
        ]
        for req in minimum_requirements_of_config:
            if not req in self._configs:
                raise ValueError('''The given configuration is not legal. It should
                    contain `chosen_evaluation`, `container_size` and `database`.''')
        if not config_fields.chosen_evaluations in self._configs:
            print('NOTICE: None of evaluation options are selected, use {} as default.'.format(
                eval_fields.facenet))
            self._configs[config_fields.chosen_evaluations] = [eval_fields.facenet]

        if not self._configs[config_fields.database]:
            print ('NOTICE: No database is assigned, some evaluations can not be executed.')
            # NOTE @kv: Should we ignore given attribute names & use category as attr_name?
        
        # Make sure self._config is legal dict before parsing.
        self._parse()


    def _parse(self):

        # Fetch valid evaluation names
        self._all_evaluation_options= self._configs[config_fields.evaluation_options]
        self._chosen_evaluation_names = self._configs[config_fields.chosen_evaluations]

        # Save evaluator system information
        self._container_size = self._configs[config_fields.container_size]
        self._database = self._configs[config_fields.database]
        
        # Parse legal evaluation applications
        self._evaluations = {}
        for _eval_name in self._chosen_evaluation_names:
            if _eval_name in REGISTERED_EVALUATION_OBJECTS and _eval_name in self._all_evaluation_options:
                self._evaluations[_eval_name] = self._all_evaluation_options[_eval_name]

        # Collect attributes defined in each evaluation
        required_attributes = []
        for _eval_name, _content in self._evaluations.items():
            if _content is None:
                continue
            if config_fields.attribute in _content:
                if not _content[config_fields.attribute] is None:
                    attrs = _content[config_fields.attribute]
                    required_attributes.extend(attrs)
        self._required_attributes = list(set(required_attributes))

        # Extract all defined names in metric standard fields
        #metric_items = inspect.getmembers(metric_fields, lambda a: not(inspect.isroutine(a)))
        #metric_items = [a for a in metric_items if not(a[0].startswith('__') and a[0].endswith('__'))]
        #self._metric_items = [v for _, v in metric_items]

    @property
    def chosen_evaluation_names(self):
        return self._chosen_evaluation_names

    @property
    def evaluations(self):
        return self._evaluations

    @property
    def database_type(self):
        return self._database

    @property
    def container_size(self):
        return self._container_size

    @property
    def required_attributes(self):
        return self._required_attributes

    def get_per_eval_config(self, eval_name):
        """
          Args:
            evaluation_name, string: Name of the evaluation object.
                e.g. MockEvaluation, same as the class.
          Return:
            per_eval_config, dict:
                Description of the per evaluation.
        """
        if eval_name in self.evaluations:
            return self.evaluations[eval_name]
        else:
            return {}

    def has_attribute(self, eval_name):
        _attrs = self.get_attributes(eval_name)

        if not _attrs or attr_fields.all_classes in _attrs:
            return False
        else:
            return True

    def get_attributes(self, eval_name=None):
        """
          Args:
            eval_name, string: Name of the evaluation object.
          Return:
            List of attributes to be evaluated at given evaluation.
        """
        if eval_name is None:
            if self._required_attributes:
                return self.required_attributes
            else:
                return [attr_fields.all_classes]

        per_eval = self.get_per_eval_config(eval_name)
        if not per_eval:
            return []
        if config_fields.attribute in per_eval:
            if not per_eval[config_fields.attribute]:
                return [attr_fields.all_classes]
            else:
                return per_eval[config_fields.attribute]
        else:
            return [attr_fields.all_classes]

    def get_metrics(self, eval_name=None):
        """
          Args:
            eval_name: Name of the evaluation object, return all metrics if not given.
          Return:
            metrics_dict: Dict of metrics and corresponding parameters.
        """
        if eval_name is None:
            all_metrics = {}
            for _eval_name in self.chosen_evaluation_names:
                if config_fields.metric in self.evaluations[_eval_name]:
                    all_metrics[_eval_name] = self.evaluations[_eval_name][config_fields.metric]
            return all_metrics

        per_eval = self.get_per_eval_config(eval_name)
        if not per_eval:
            return {}
        # return the dict which key in metric standard fields
        metrics_dict = {}
        if config_fields.metric in per_eval:
            metrics_dict = per_eval[config_fields.metric]
        return metrics_dict
