import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from metric_learning_evaluator.config_parser.standard_fields import ConfigStandardFields as config_fields
from metric_learning_evaluator.query.zeus_wrapper import ZeusWrapper
from metric_learning_evaluator.query.json_wrapper import JsonWrapper
from metric_learning_evaluator.query.native_wrapper import NativeWrapper

DATABASE_DICT = {
    config_fields.native: NativeWrapper,
}


class QueryInterface(object):

    def __init__(self, database_setting):
        """
        database_setting:
            reference config.database
        """
        database_type = database_setting.get(config_fields.database_type, None)
        database_config = database_setting.get(config_fields.database_config, None)
        if not database_type in DATABASE_DICT:
            raise ValueError('Given database type is not support.')
        self.database_type = database_type

        # create wrapper
        database_object = DATABASE_DICT[database_type]
        self.database = database_object(database_config)
        print('Initialize Query Interface with {} backend.'.format(database_type))

    def query(self, instance_id, required_attribute_names=None):
        """Query function:

            Args:
                instance_id, int:
                    Unique index describing given image.

                required_attribute_names, list of strings:
                    Kind of filtering condition for querying database.
            
            Return:
                attributes: list of strings:
                    Attributes for given instance_id.

            Usage examples:

            Database:
                instance_id: 12, with three attributes: Color.Red, Shape.Bottle, Pose.isFront

            Query example with instance_id = 12:
                attribute_names: Color
                return: attribute_values: Color.Red

                attribute_names: Color.Blue
                return: attribute_values: []

                attribute_names: AllAttributes or None
                return: attribute_values: Color.Red, Shape.Bottle, Pose.isFront

                attribute_names: Color, Shape
                return: attribute_values: Color.Red, Shape.Bottle

                attribute_names: Color, Shape.Can
                return: attribute_values: "Color.Red" (list of str)
        """

        all_attr_names = self.database.query_attributes_by_instance_id(instance_id)

        if required_attribute_names is None:
            return all_attr_names
        else:
            return list(set(all_attr_names) & set(required_attribute_names))
