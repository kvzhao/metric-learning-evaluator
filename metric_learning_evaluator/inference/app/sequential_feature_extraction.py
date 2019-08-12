"""
  TODO: Change the logic
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

from metric_learning_evaluator.inference.utils.image_utils import read_jpeg_image
from metric_learning_evaluator.inference.utils.read_json_labelmap import read_json_labelmap

from metric_learning_evaluator.utils.sample_strategy import SampleStrategy

from metric_learning_evaluator.inference.components.extractor import FeatureExtractor

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.data_tools.feature_object import FeatureObject

from metric_learning_evaluator.core.standard_fields import SampleStrategyStandardFields as sample_fields


def extraction_application(configs, args):

    # parse static configs
    model_configs = configs['extractor_settings']
    labelmap_path = configs['labelmap_path']
    container_capacity = configs['embedding_container_capacity']

    img_size = model_configs['image_size']
    embedding_size = model_configs['embedding_size']
    model_path = model_configs['model_path']
    database_path = model_configs['database_path']

    # parse arguments
    data_dir = args.data_dir
    out_dir = args.out_dir
    data_type = args.data_type

    ## ===== Error Proofing =======
    if data_dir is None:
        raise ValueError('data_dir is not given')
    if out_dir is None:
        out_dir = '/tmp/extracted_features'
        print('NOTEICE: out_dir is not given, save to {} in default'.format(out_dir))

    if labelmap_path is not None:
        # label maps
        labelmap = read_json_labelmap(labelmap_path)
        category_name_map = {} # not used
        for _, content in labelmap.items():
            category_name_map[content['unique_id']] = content['label_name']
    else:
        # create own labelmap
        print('NOTICE: labelmap is not given,')
        labelmap = {}

    # Restore frozen model
    embedder = FeatureExtractor(model_path, img_size)

    ### =================================
    ###  Sequentially extract embeddings
    ### =================================
    all_image_filenames = []
    if data_type == 'datasetbackbone':
        print('Extract features from dataset backbone')
        dataset_backbone_db_path = glob.glob(data_dir + '/*.db')

        # TODO: check whether the path is legal or not
        src_db = DatasetBackbone(data_dir)
        filenames = src_db.query_all_img_filenames() # can be used as instance_ids

        label_ids, instance_ids = [], []
        embedding_container = EmbeddingContainer(embedding_size, 0, container_capacity)
        try:
            for filename in tqdm(filenames):
                """TODO: cropped or not?"""
                annotation = src_db.query_anno_info_by_filename(filename)[0]
                instance_id = annotation['id']
                category_name = annotation['category']
                if not category_name in labelmap:
                    #print('NOTICE: {} cannot be found in {} which would be skipped'.format(category_name, labelmap_path))
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
        except:
            """TODO:
                Save embeddings up to current state
            """
            print('Exception happens for ? reason.')
            pass

    elif data_type == 'folder':
        # folder contains raw images
        input_filenames = [
            join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
        print('Load input data from folder with {} images.'.format(len(input_filenames)))

    feature_exporter = FeatureObject()
    feature_exporter.filename_strings = np.asarray(filenames)
    feature_exporter.embeddings = embedding_container.embeddings
    print('Export embedding with shape: {}'.format(embedding_container.embeddings.shape))

    # suppose the groundtruth is provided
    if data_type == 'datasetbackbone':
        category_names, category_ids = [], []
        try:
            for name in all_image_filenames:
                annos = src_db.query_anno_info_by_filename(name)
                for anno in annos:
                    cate_name = anno['category']
                    if not cate_name in labelmap:
                        category_names.append('unknown_instance')
                        category_ids.append(anno['cate_id'])
                    else:
                        category_names.append(labelmap[cate_name]['label_name'])
                        category_ids.append(labelmap[cate_name]['unique_id'])
        except KeyboardInterrupt:
            print('Interrupted by user, save {} features to {}'.format(len(category_names)))
            feature_exporter.label_names = np.asarray(category_names)
            feature_exporter.label_ids = np.asarray(category_ids)
            feature_exporter.save(out_dir)

        if category_names:
            feature_exporter.label_names = np.asarray(category_names)
        if category_ids:
            feature_exporter.label_ids = np.asarray(category_ids)

    feature_exporter.save(out_dir)
    print("Save all extracted features at {}".format(out_dir))