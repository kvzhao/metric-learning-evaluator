import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../..')))

from os import listdir
from os.path import join, isfile

import cv2
import numpy as np

import yaml

import scutils

from metric_learning_evaluator.inference.app.map_labeling_to_detection import mapping_application 

def main(args):
    with open(args.config_path, "r") as f:
        config = yaml.load(f)
    
    mapping_application(config, args)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Two Stage Inference Tool')

    parser.add_argument('-c', '--config_path', type=str, default='two_stage_config.yml',
                        help='Path to the configuration yaml file.')
    parser.add_argument('-dd', '--data_dir', type=str, default=None,
                        help='Path to Input DatasetBackbone or raw image folder.')
    parser.add_argument('-od', '--out_dir', type=str, default=None,
                        help='Path to Output DatasetBackbone.')
    parser.add_argument('-mit', '--matching_iou_threshold', type=float, default=None,
                        help='Threshold of IOU matching.')
    
    args = parser.parse_args()
    main(args)
