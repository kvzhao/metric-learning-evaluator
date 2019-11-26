import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import re
import yaml

from metric_learning_evaluator.core.standard_fields import ConfigStandardFields as config_fields
from metric_learning_evaluator.core.standard_fields import EvaluationStandardFields as eval_fields
from metric_learning_evaluator.core.standard_fields import AttributeStandardFields as attr_fields
from metric_learning_evaluator.core.registered import REGISTERED_EVALUATION_OBJECTS
from metric_learning_evaluator.core.registered import REGISTERED_DATABASE_TYPE
from metric_learning_evaluator.core.registered import REGISTERED_INDEX_AGENT


class ConfigParser(object):
    """Evaluator Configuration:
      User Scenario:
        1. load from given config path
        2. get useful values to set containers
        3. get hyper-parameters for evaluation functions
      Principal and Convention of using configurations:
    """

    def __init__(self, config_dict):
        """
          Args:
            configs: A dictionary loaded from eval_config.yaml
        """

        self._configs = config_dict

        minimum_requirements_of_config = [
            config_fields.container_size,
            config_fields.evaluation_options,
            config_fields.database,
        ]

        for req in minimum_requirements_of_config:
            if req not in self._configs:
                raise ValueError('''The given configuration is not legal. It should
                    contain `chosen_evaluation`, `container_size` and `database`.''')
        if config_fields.chosen_evaluations not in self._configs:
            print('NOTICE: None of evaluation options are selected, use {} as default.'.format(
                eval_fields.facenet))
            self._configs[config_fields.chosen_evaluations] = [eval_fields.facenet]

        if not self._configs[config_fields.database]:
            print('NOTICE: No database is assigned, some evaluations can not be executed.')
            # NOTE @kv: Should we ignore given attribute names & use category as attr_name?

        # Make sure self._config is legal dict before parsing.
        self._parse()

    def _parse(self):

        # Fetch valid evaluation names
        self._all_evaluation_options = self._configs[config_fields.evaluation_options]
        self._chosen_evaluation_names = self._configs[config_fields.chosen_evaluations]

        # Save evaluator system information
        self._container_size = self._configs[config_fields.container_size]
        self._database = self._configs[config_fields.database]
        self._index_agent = self._configs[config_fields.index_agent]

        # Index Agent is Numpy by default
        if self._index_agent not in [config_fields.numpy_agent, config_fields.hnsw_agent]:
            self._index_agent = config_fields.numpy_agent

        # Parse legal evaluation applications
        self._evaluations = {}

        for _eval_name in self._chosen_evaluation_names:
            if _eval_name in REGISTERED_EVALUATION_OBJECTS and _eval_name in self._all_evaluation_options:
                self._evaluations[_eval_name] = EvaluationConfigParser(
                    self._all_evaluation_options[_eval_name],
                    self.has_query_interface,
                    self.index_agent)

        # Collect attribute items defined in each evaluation
        required_attributes = []
        for _eval_name, _eval_config in self._evaluations.items():
            required_attributes.extend(_eval_config.attribute_items)
        self._required_attributes = list(set(required_attributes))

    @property
    def chosen_evaluation_names(self):
        # return: list of string
        return self._chosen_evaluation_names

    @property
    def evaluations(self):
        # return: dict of objects (EvaluationConfigParser)
        return self._evaluations

    @property
    def database(self):
        # return: dict of database section
        return self._database

    @property
    def has_query_interface(self):
        # return: A boolean
        if self.database is None:
            return False
        db_type = self.database[config_fields.database_type]
        if db_type is None:
            return False
        elif db_type not in REGISTERED_DATABASE_TYPE:
            return False
        else:
            return True

    @property
    def container_size(self):
        # return: an integer
        return self._container_size

    @property
    def required_attributes(self):
        """List of each individual attribute key
        """
        return self._required_attributes

    @property
    def index_agent(self):
        # return: A string
        return self._index_agent

    # TODO: Change name:
    def get_eval_config(self, eval_name):
        """
          Args:
            eval_name, string: Name of the evaluation object.
                e.g. MockEvaluation, same as the class.
          Return:
            An object: evaluation config parser.
        """
        if eval_name in self.evaluations:
            return self.evaluations[eval_name]
        else:
            return None

    # TODO @kv: Deprecate this function
    def has_attribute(self, eval_name=None):
        # Rule: If no database is given, no attribute can be obtained
        if not self.has_query_interface:
            return False
        _attrs = self.get_attributes(eval_name)
        if not _attrs:
            return False
        else:
            return True

    def get_attributes(self, eval_name=None):
        # get all attributes commands, remove duplications
        commands = []
        commands.extend(self.get_attribute_cross_reference_commands(eval_name))
        commands.extend(self.get_attribute_group_commands(eval_name))
        return list(set(commands))

    def get_attribute_cross_reference_commands(self, eval_name=None):
        """
          Args:
            eval_name: string
          Return:
            A list of commands
        """
        if eval_name is None:
            all_cross_reference_commands = []
            for name, eval_config in self.evaluations.items():
                all_cross_reference_commands.extend(
                    eval_config.attribute_cross_reference_commands)
            return list(set(all_cross_reference_commands))
        elif eval_name in self.evaluations:
            return self.evaluations[eval_name].attribute_cross_reference_commands
        else:
            return []

    def get_attribute_group_commands(self, eval_name=None):
        """
          Args:
            eval_name: string
          Return:
            A list of commands
        """
        if eval_name is None:
            all_group_commands = []
            for name, eval_config in self.evaluations.items():
                all_group_commands.extend(eval_config.attribute_group_commands)
            return list(set(all_group_commands))
        elif eval_name in self.evaluations:
            return self.evaluations[eval_name].attribute_group_commands
        else:
            return [attr_fields.All]

    def get_attribute_items(self, eval_name=None):
        """
          Args:
            eval_name: string, Name of the evaluation object.
            attr_type: string. Two types of attributes: 1. cross_reference, 2. group
          Return:
            attribute: list of string, key of the evaluation items assigned in config.
          NOTE (Important!) `required_attributes` are not same as `attributes`
        """
        attr_items = []
        if eval_name is None:
            for name, eval_config in self.evaluations.items():
                attr_items.extend(eval_config.attribute_items)
        elif eval_name in self.evaluations:
            attr_items.extend(self.evaluations[eval_name].attr_items)
        return attr_items

    # TODO: Use EvaluationConfigParser
    def get_metrics(self, eval_name=None):
        """
          Args:
            eval_name: Name of the evaluation object, return all metrics if not given.
          Return:
            metrics_dict: dict or dict of dict
                - dict of metric section 
                - dict of dict (eval_name to attribute_section)
        """
        if eval_name is None:
            all_metrics = {}
            for _name, _eval_config in self.evaluations.items():
                all_metrics[_name] = _eval_config.metric_section
            return all_metrics
        elif eval_name in self.evaluations:
            return self.evaluations[eval_name].metric_section
        else:
            return {}
        # return the dict which key in metric standard fields
        metrics_dict = {}
        if config_fields.metric in per_eval:
            metrics_dict = per_eval[config_fields.metric]
        return metrics_dict


class EvaluationConfigParser(object):
    def __init__(self, per_eval_dict, has_database=False, agent_type=config_fields.numpy_agent):
        """Per Evaluation Config

           This object will be passed to each EvaluationObject which handles:
            - sampling
            - metric
            - distance_measure
            - attribute
            - option

           And some other consistent condition like
            - has database
           NOTE:
            If no query database is provided, no attributes can be queired.
        """
        self._eval_config = per_eval_dict
        self._has_database = has_database
        self._agent_type = agent_type

    @property
    def has_database(self):
        return self._has_database

    @property
    def agent_type(self):
        return self._agent_type

    @property
    def sampling_section(self):
        if config_fields.sampling in self._eval_config:
            return self._eval_config[config_fields.sampling]
        else:
            return {}

    @property
    def distance_measure(self):
        if config_fields.distance_measure in self._eval_config:
            return self._eval_config[config_fields.distance_measure]
        else:
            return {}

    @property
    def metric_section(self):
        if config_fields.metric in self._eval_config:
            return self._eval_config[config_fields.metric]
        else:
            return {}

    @property
    def attribute_section(self):
        # return: A dict
        _attr = self._eval_config.get(config_fields.attribute, None)
        if _attr is None:
            return {}
        return _attr

    @property
    def attributes(self):
        # flatten list of string
        commands = []
        commands.extend(self.attribute_cross_reference_commands)
        commands.extend(self.attribute_group_commands)
        commands = list(set(commands))
        if not commands:
            commands.append(attr_fields.All)
        return commands

    @property
    def attribute_items(self):
        # return: List of string
        # individual attribute name used as key in container
        attr_items = []
        attr_items.extend(self.attribute_group_items)
        attr_items.extend(self.attribute_cross_reference_items)
        attr_items = list(set(attr_items))
        if not attr_items:
            attr_items.append(attr_fields.All)
        return attr_items

    @property
    def attribute_cross_reference_commands(self):
        # return: List of commands
        attr_section = self.attribute_section
        if config_fields.cross_reference in attr_section and attr_section[config_fields.cross_reference]:
            # parse string respectively
            cross_reference_commands = attr_section[config_fields.cross_reference]
            cross_reference_commands = [self._remove_blanks(cmd)
                for cmd in cross_reference_commands if cmd is not None]
            return list(set(cross_reference_commands))
        return []

    @property
    def attribute_group_items(self):
        # return: List of strings (attr names)
        group_commands = self.attribute_group_commands
        if group_commands:
            attr_items = []
            for cmd in group_commands:
                attr_items.extend(
                    self._parse_single_attribute_command(cmd))
            return list(set(attr_items))
        else:
            return [attr_fields.All]

    @property
    def attribute_group_commands(self):
        # return: List of commands
        attr_section = self.attribute_section
        if config_fields.group in attr_section and attr_section[config_fields.group]:
            # parse string respectively
            group_commands = attr_section[config_fields.group]
            group_commands = [self._remove_blanks(cmd) for cmd in group_commands if cmd is not None]
            return list(set(group_commands))
        return [attr_fields.All]

    @property
    def attribute_cross_reference_items(self):
        # return: List of strings (attr names)
        cross_reference_commands = self.attribute_cross_reference_commands
        if cross_reference_commands:
            attr_items = []
            for cmd in cross_reference_commands:
                src_cmd, tar_cmd = self._split_cross_reference_command(cmd)
                attr_items.extend(
                    self._parse_single_attribute_command(src_cmd))
                attr_items.extend(
                    self._parse_single_attribute_command(tar_cmd))
            return attr_items
        return []

    @property
    def option_section(self):
        if config_fields.option in self._eval_config:
            return self._eval_config[config_fields.option]
        return {}

    @staticmethod
    def _parse_single_attribute_command(command):
        # parse attribute single line command
        command = command.replace(' ', '')
        command = re.sub(r'[(){}]', '', command)
        attrs = re.split(r'\+|\-', command)
        return attrs

    @staticmethod
    def _split_cross_reference_command(command):
        m = re.match(r'(.+)->(.+)', command)
        source = m.group(1)
        target = m.group(2)
        return source, target

    @staticmethod
    def _remove_blanks(string):
        return string.replace(' ', '')
