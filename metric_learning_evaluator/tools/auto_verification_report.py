
import os
import sys
import yaml
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from metric_learning_evaluator.deploy.auto_verification import AutoVerification


def main(args):
    data_dir = args.data_dir
    sys_config_path = args.sys_config_path
    force_update = args.force_update

    with open(sys_config_path, 'r') as fp:
        config_dict = yaml.load(fp)

    auto_verification = AutoVerification(config_dict=config_dict,
                                         verbose=True)

    # load embedding db from disk
    auto_verification.load(data_dir)

    auto_verification.preprocessing(force_update=force_update,
                                    save=True)

    run_config = {}
    auto_verification.run(run_config)

    select_config = {}
    # default: return all anchors
    ids = auto_verification.anchor_selection(select_config)
    print('Select #of anchors = {}'.format(len(ids)))

    report_config = {}
    report = auto_verification.report(report_config, to_dict=False)
    print(report)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Auto Verification Tool')

    parser.add_argument('-dd', '--data_dir', type=str, default=None,
                        help='Path to EmbeddingDB pkl path')
    parser.add_argument('-sc', '--sys_config_path', type=str, default=None,
                        help='Configuration for verification measures')
    parser.add_argument('-f', '--force_update', action='store_true')

    args = parser.parse_args()
    main(args)
