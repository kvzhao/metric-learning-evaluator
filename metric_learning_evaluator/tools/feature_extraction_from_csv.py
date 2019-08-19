
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))  # noqa

from os import listdir
from os.path import join, isfile

import cv2
import math
import yaml
import csv
import glob

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from metric_learning_evaluator.inference.utils.image_utils import read_jpeg_image
from metric_learning_evaluator.query.csv_reader import CsvReader
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.core.standard_fields import AttributeTableStandardFields as table_fields


def main(args):
    config_path = args.config_path
    csvfile_path = args.csv_file
    output_dir = args.out_dir
    extractor = args.extractor_type

    assert extractor == 'f' or extractor == 'm' , 'choose a suitable feature extractor. ("f" for facenet , "m" for metric-learning.)'
    if extractor == 'f':
       from metric_learning_evaluator.inference.components.facenet_extractor import FeatureExtractor
    elif extractor == 'm':
       from metric_learning_evaluator.inference.components.extractor import FeatureExtractor

    with open(config_path, 'r') as f:
        configs = yaml.load(f)

    model_configs = configs['extractor_settings']
    container_capacity = configs['embedding_container_capacity']
    img_size = model_configs['image_size']
    embedding_size = model_configs['embedding_size']
    model_path = model_configs['model_path']

    image_filenames = []
    instance_ids = []
    label_ids = []
    label_names = []
    with open(csvfile_path, newline='', encoding="utf-8") as csv_file:
        csv_rows = csv.DictReader(csv_file)
        for row in csv_rows:
            image_filenames.append(row[table_fields.image_path])
            instance_ids.append(row[table_fields.instance_id])
            label_ids.append(row[table_fields.label_id])
            label_names.append(row[table_fields.label_name])
    attr_reader = CsvReader({'path': csvfile_path})

    total_number = len(image_filenames)
    container = EmbeddingContainer(embedding_size=embedding_size,
                                   container_size=total_number)
    embedder = FeatureExtractor(model_path, img_size)

    try:
        for inst_id, img_path, label_id, label_name in tqdm(zip(instance_ids, image_filenames, label_ids, label_names)):
            attributes = attr_reader.query_attributes_by_instance_id(inst_id)
            try:
                image = read_jpeg_image(img_path, img_size)
            except:
                print('image: {} reading fails, skip'.format(img_path))
                continue
            try:
                feature = embedder.extract_feature(image)
            except:
                print('feature extraction: {} reading fails, skip'.format(img_path))
                continue
            container.add(inst_id, label_id, feature,
                          attributes=attributes,
                          label_name=label_name,
                          filename=img_path)
    except:
        # Stop temporarily
        print('Stop unexpectedly, save the intermediate results.')
        container.save(output_dir)
    container.save(output_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Evaluation for Given Database')

    parser.add_argument('-e', '--extractor_type', type=str, default=None,
                        help='f for facenet pb, m for metric-learning pb')

    parser.add_argument('-c', '--config_path', type=str, default=None)

    parser.add_argument('-csv', '--csv_file', type=str, default=None,
                        help='')
    parser.add_argument('-od', '--out_dir', type=str, default=None,
                        help='')

    args = parser.parse_args()
    main(args)
