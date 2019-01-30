
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from core.eval_standard_fields import ConfigStandardFields as config_fields
from query.datasetbackbone_wrapper import DatasetBackboneWrapper
from query.zeus_wrapper import ZeusWrapper

DATABASE_DICT = {
    config_fields.datasetbackbone: DatasetBackboneWrapper,
    config_fields.zeus: ZeusWrapper,
}

class QueryInterface(object):
    
    def __init__(self, database_type):
        """
        """
        if not database_type in DATABASE_DICT:
            raise ValueError('Given database type is not support.')
        self.database_type = database_type

        self.backend_db = DATABASE_DICT[database_type]

    def query(self, image_id, required_attribute_names):
        """Query function:

            Args:
                image_id, int:
                    Unique index describing given image.

                required_attribute_names, list of strings:
                    Kind of filtering condition for querying database.
            
            Return:
                attributes: list of strings:
                    Attributes for given image_id.

            Usage examples:

            Database:
                image_id: 12, with three attributes: Color.Red, Shape.Bottle, Pose.isFront

            Query example with image_id = 12:
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

        return ["MOCK_QUERIED_RESULTS", "MOCK_FETCHED_ATTRIBUTES"]