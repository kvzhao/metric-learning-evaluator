import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from metric_learning_evaluator.core.standard_fields import ConfigStandardFields as config_fields
from metric_learning_evaluator.core.standard_fields import QueryDatabaseStandardFields as db_fields
# NOTE: Change the default (& the only) supported database format.
from metric_learning_evaluator.core.registered import REGISTERED_DATABASE_TYPE


class QueryInterface(object):

    def __init__(self, database_setting):
        """
        database_setting:
            reference config.database
        """
        database_type = database_setting.get(config_fields.database_type, None)
        database_config = database_setting.get(config_fields.database_config, None)
        if database_type not in REGISTERED_DATABASE_TYPE:
            raise ValueError('Given database type is not support.')
        self.database_type = database_type

        # create wrapper
        database_object = REGISTERED_DATABASE_TYPE[database_type]
        self.database = database_object(database_config)
        print('Initialize Query Interface with {} backend.'.format(database_type))

    def query(self, instance_id):
        """Query function:

            Args:
                instance_id, int:
                    Unique index describing given image.

                required_attribute_names, list of strings:
                    Kind of filtering condition for querying database.
                NOTE: This argument is removed for current stage.

            Return:
                attributes: A dictionary with attribute name & attribute value
                            e.g. {color: red, size: 150}
        """
        return self.database.query_attributes_by_instance_id(instance_id)
