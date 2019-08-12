"""
  This script is used for testing given csv file.
"""

from metric_learning_evaluator.query.general_database import QueryInterface
from metric_learning_evaluator.data_tools.attribute_table import AttributeTable


def main(args):
    database_setting = {
        'database_type': 'CSV',
        'database_config': {
            'path': args.csv_path,
        }
    }
    qi = QueryInterface(database_setting)

    table = AttributeTable()

    # hahaha
    for idx in range(500000):
        attrs = qi.query(idx)
        if attrs:
            table.add(idx, attrs)

    print(table.DataFrame)
    name = 'attribute_table.csv'
    if args.output_name:
        name = args.output_name
    table.save(name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv_path', type=str, default=None,
                        help='Path to given csv file')
    parser.add_argument('-o', '--output_name', type=str, default=None,
                        help='Output name of csv file')

    args = parser.parse_args()
    main(args)
