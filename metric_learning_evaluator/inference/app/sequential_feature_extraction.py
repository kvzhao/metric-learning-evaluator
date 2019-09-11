"""
    Input
        - folder
        - csv
        - datasetbackbone
    Output
        - EmbeddingContainer

    Input situation
        - 

    Labelmap format
"""

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

from scutils.scdata import DatasetBackbone
from metric_learning_evaluator.query.csv_reader import CsvReader

from metric_learning_evaluator.inference.utils.image_utils import read_jpeg_image
from metric_learning_evaluator.inference.utils.read_json_labelmap import read_json_labelmap

from metric_learning_evaluator.inference.utils.image_utils import load_file_from_folder
from metric_learning_evaluator.inference.utils.image_utils import load_file_from_structure_folder

from metric_learning_evaluator.inference.components.extractor import FeatureExtractor
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.core.standard_fields import AttributeTableStandardFields as table_fields


"""TODO:
    Change the application interface to
        custom_application(configs, datasets)
"""

SUPPORTED_DATATYPES = [
    'csv',
    'folder',
    'datasetbackbone',
]


def extraction_application(configs, args):

    # parse static configs
    model_configs = configs['extractor_settings']
    labelmap_path = configs['labelmap_path']
    container_capacity = configs['embedding_container_capacity']

    img_size = model_configs['image_size']
    embedding_size = model_configs['embedding_size']
    model_path = model_configs['model_path']

    # TODO: Check if None or not
    database_path = model_configs['database_path']

    # parse arguments
    data_dir = args.data_dir
    out_dir = args.out_dir
    data_type = args.data_type

    # ===== Error Proofing =======
    if data_dir is None:
        raise ValueError('data_dir is not given')
    if out_dir is None:
        out_dir = '/tmp/extracted_features'
        print('NOTICE: out_dir is not given, save to {} in default'.format(out_dir))

    # TODO: remove labelmap
    labelmap = None
    if labelmap_path is not None:
        # label maps
        labelmap = read_json_labelmap(labelmap_path)
        category_name_map = {}
        for _, content in labelmap.items():
            category_name_map[content['unique_id']] = content['label_name']

    # Restore frozen model
    embedder = FeatureExtractor(model_path, img_size)

    # =================================
    # Data preprocessing
    # =================================

    # all input information
    attributes = None
    instance_ids = None
    image_filenames, label_names, label_ids = None, None, None

    if data_type == 'datasetbackbone':
        print('Extract features from dataset backbone')
        # For sanity check
        dataset_backbone_db_path = glob.glob(data_dir + '/*.db')

        # TODO: check whether the path is legal or not
        # We should handle dataset backbone boxes!
        src_db = DatasetBackbone(data_dir)
        short_filenames = src_db.query_all_img_filenames()
        image_filenames = src_db.query_img_abspath_by_filename(short_filenames)
        # can be used as instance_ids

        # prepare label ids and names

        label_ids, instance_ids = [], []
        try:
            pass
            """
            for filename in tqdm(filenames):
                annotation = src_db.query_anno_info_by_filename(filename)[0]
                instance_id = annotation['id']
                category_name = annotation['category']
                if category_name not in labelmap:
                    # print('NOTICE: {} cannot be found in {} which would be skipped'.format(category_name, labelmap_path))
                    label_id = annotation['cate_id']
                    label_ids.append(label_id)
                else:
                    instance_ids.append(filename)
                    label_id = labelmap[category_name]['unique_id']
                    label_ids.append(label_id)
                img_path = src_db.query_img_abspath_by_filename(filename)[0]
                img = read_jpeg_image(img_path, img_size)
                feature = embedder.extract_feature(img)
                embedding_container.add(instance_id=instance_id,
                                        label_id=label_id,
                                        embedding=feature[0])
                # available filename in same order
                all_image_filenames.append(filename)
            """
        except:
            pass
    elif data_type == 'folder':
        # folder contains raw images
        image_filenames, label_names = load_file_from_structure_folder(data_dir)
        # TODO: Create label id itself
    elif data_type == 'csv':
        csvfile_path = data_dir
        instance_ids = []
        attributes = []
        image_filenames, label_names, label_ids = [], [], []

        with open(csvfile_path, newline='', encoding="utf-8") as csv_file:
            csv_rows = csv.DictReader(csv_file)
            # fetch necessary
            for line_id, row in enumerate(csv_rows):
                inst_id = row.get(table_fields.instance_id, line_id)
                instance_ids.append(inst_id)
                if table_fields.instance_id in row:
                    row.pop(table_fields.instance_id)
                if table_fields.image_path in row:
                    image_filenames.append(row.pop(table_fields.image_path))
                if table_fields.label_id in row:
                    label_ids.append(row.pop(table_fields.label_id))
                if table_fields.label_name in row:
                    label_names.append(row.pop(table_fields.label_name))
                # Rest of them are attributes (and remove noises)
                row.pop('', None)
                attributes.append(row)

    if labelmap is None:
        labelmap = {}
    if label_ids is None:
        label_name_set = set(label_names)
        labelmap = {name: index
                    for index, name in enumerate(label_name_set)}
        label_ids = [labelmap[name] for name in label_names]

    if instance_ids is None:
        instance_ids = [idx for idx in range(len(image_filenames))]
    if attributes is None:
        attributes = [{}] * len(image_filenames)

    # =================================
    # Feature extraction loop
    # =================================
    num_total_images = len(image_filenames)
    container = EmbeddingContainer(embedding_size=embedding_size,
                                   container_size=num_total_images)
    try:
        for inst_id, img_path, label_id, label_name, attr_dict in tqdm(
            zip(instance_ids, image_filenames, label_ids, label_names, attributes)):
            try:
                image = read_jpeg_image(img_path, img_size)
            except:
                print('image: {} reading fails, skip'.format(img_path))
                continue
            try:
                feature = embedder.extract_feature(image)
                feature = np.squeeze(feature)
            except:
                print('feature extraction: {} inference fails, skip'.format(img_path))
                continue
            container.add(inst_id,
                          label_id,
                          feature,
                          attribute=attr_dict,
                          label_name=label_name,
                          filename=img_path)
    except:
        # Stop temporarily
        print('Stop unexpectedly, save the intermediate results.')
        container.save(out_dir)
    container.save(out_dir)
    print("Save all extracted features at {}".format(out_dir))
