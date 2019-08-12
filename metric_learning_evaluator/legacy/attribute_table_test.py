
from metric_learning_evaluator.data_tools.attribute_table import AttributeTable


def main(args):
    attr_table = AttributeTable(args.data_dir)

    """
      Push instances
      
      property
        color := {red, blue}
        shape := {square, round}
      domain
        {query, database}
        {seen, unseen}
    """

    attr_table.insert_domain(0, 'database')
    attr_table.insert_domain(1, 'database')
    attr_table.insert_domain(2, 'database')
    attr_table.insert_domain(3, 'database')
    attr_table.insert_domain(4, 'database')
    attr_table.insert_domain(5, 'query')
    attr_table.insert_domain(6, 'query')
    attr_table.insert_domain(7, 'query')
    attr_table.insert_domain(8, 'query')
    attr_table.insert_domain(9, 'query')

    attr_table.insert_property(0, 'color', 'red')
    attr_table.insert_property(1, 'color', 'red')
    attr_table.insert_property(2, 'color', 'red')
    attr_table.insert_property(3, 'color', 'red')
    attr_table.insert_property(4, 'color', 'blue')
    attr_table.insert_property(5, 'color', 'blue')
    attr_table.insert_property(6, 'color', 'blue')

    attr_table.insert_property(0, 'shape', 'round')
    attr_table.insert_property(1, 'shape', 'round')
    attr_table.insert_property(2, 'shape', 'round')
    attr_table.insert_property(3, 'shape', 'round')
    attr_table.insert_property(4, 'shape', 'round')
    attr_table.insert_property(5, 'shape', 'round')
    attr_table.insert_property(6, 'shape', 'square')
    attr_table.insert_property(7, 'shape', 'round')
    attr_table.insert_property(8, 'shape', 'round')
    attr_table.insert_property(9, 'shape', 'square')


    attr_table.commit()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Create testable attribute table')

    parser.add_argument('-dd', '--data_dir', type=str, default=None,
                        help='Path to query feature object')

    args = parser.parse_args()

    main(args)