"""
The program extract embeddings and label ids & names as search database.

Scenario
    - From dataset backbone:
        Images in backbone are items with correct annotations.
        NOTE: The given dataset backbone should be cropped.
    - From folder:
        These images are not annotated, used as `new` categories.

TODO:
    - augmentation: rotate the db categories
"""

import csv
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))  # noqa

import cv2
import math
import yaml
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.image_utils import read_jpeg_image
from utils.read_json_labelmap import read_json_labelmap

from metric_learning_evaluator.utils.sample_strategy import SampleStrategy
from metric_learning_evaluator.utils.sample_strategy import sample_fields

from metric_learning_evaluator.inference.components.extractor import FeatureExtractor

from metric_learning_evaluator.data_tools.image_container import ImageContainer
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.data_tools.feature_object import FeatureObject



def main(args):
    # NOTE: Modify exported_dir to the folder including save_model.pb

    # model
    model_dir = args.model_dir

    # data
    data_dir = args.data_dir
    out_dir = args.out_dir
    data_type = args.data_type
    img_size = args.img_size
    #batch_size = args.batch_size
    labelmap_path = args.labelmap

    ## path error proofing
    if data_dir is None:
        raise ValueError('data_dir is not given')
    if out_dir is None:
        out_dir = '/tmp/extracted_features'
        print('NOTEICE: out_dir is not given, save to {} in default'.format(out_dir))

    # embedding
    embedding_size = args.embedding_size
    container_capacity = args.container_capacity

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
    embedder = FeatureExtractor(model_dir, img_size)

    ### =================================
    ###  Sequentially extract embeddings
    ### =================================
    all_image_filenames = []
    if data_type == 'datasetbackbone':
        from scutils.scdata import DatasetBackbone
        dataset_backbone_db_path = glob.glob(data_dir + '/*.db')

        # TODO: check whether the path is legal or not

        src_db = DatasetBackbone(data_dir)
        filenames = src_db.query_all_img_filenames() # can be used as instance_ids

        label_ids = []
        instance_ids = []
        embedding_container = EmbeddingContainer(embedding_size, 0, container_capacity)

        try:
            for filename in tqdm(filenames):
                annotation = src_db.query_anno_info_by_filename(filename)[0]
                instance_id = annotation['id']
                category_name = annotation['category']
                if not category_name in labelmap:
                    #print('NOTICE: {} cannot be found in {} which would be skipped'.format(category_name, labelmap_path))
                    label_id = annotation['cate_id']
                else:
                    instance_ids.append(filename)
                    label_id = labelmap[category_name]['unique_id']
                    label_ids.append(label_id)
                img_path = src_db.query_img_abspath_by_filename(filename)[0]
                img = read_jpeg_image(img_path, img_size)
                feature = embedder.extract_feature(img)

                embedding_container.add(instance_id=instance_id,
                                        label_id=label_id, embedding=feature[0])
                # available filename in same order
                all_image_filenames.append(filename)
        except:
            """TODO:
                Save embeddings up to current state
            """
            pass

    elif data_type == 'folder':
        pass

    feature_exporter = FeatureObject()
    feature_exporter.filename_strings = np.asarray(filenames)
    feature_exporter.embeddings = embedding_container.embeddings
    print('Export embedding with shape: {}'.format(embedding_container.embeddings.shape))

    if data_type == 'datasetbackbone':
        category_names = []
        category_ids = []
        try:
            for name in all_image_filenames:
                annos = src_db.query_anno_info_by_filename(name)
                for anno in annos:
                    cate_name = anno['category']
                    if not cate_name in labelmap:
                        # TODO:
                        print ('WARNING: {} not in labelmap,'.format(cate_name))
                        continue
                    category_names.append(labelmap[cate_name]['label_name'])
                    category_ids.append(labelmap[cate_name]['unique_id'])
        except KeyboardInterrupt:
            print('Interrupted by user, save {} features to {}'.format(len(category_names)))
            feature_exporter.label_names = np.asarray(category_names)
            feature_exporter.label_ids = np.asarray(category_ids)
            feature_exporter.save(out_dir)

        feature_exporter.label_names = np.asarray(category_names)
        feature_exporter.label_ids = np.asarray(category_ids)

    feature_exporter.save(out_dir)
    print("Save all extracted features at {}.".format(out_dir))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('Export embeddings in unified format for evaluator.')
    #parser.add_argument('-c', '--config', type=str, default='config.yml',
    #                    help='Path to the configuration yaml file.')
    parser.add_argument('-md', '--model_dir', type=str, default=None,
                        help='Path to the inference model in pb format.')
    parser.add_argument('-dd', '--data_dir', type=str, default=None,
                        help='Path base to the image data directory.')
    parser.add_argument('-dt', '--data_type', type=str, default='datasetbackbone',
                        help='Type of given data: datasetbackbone | folder .')
    parser.add_argument('-od', '--out_dir', type=str, default='extracted_embeddings',
                        help='Directory for saveing extracted embeddings.')
    parser.add_argument('-lm', '--labelmap', type=str, default=None,
                        help='Path to the label map.')
    parser.add_argument('-is', '--img_size', type=int, default=150,
                        help='Image size for predefined input placeholder.')
    parser.add_argument('-es', '--embedding_size', type=int, default=2048,
                        help='Dimension of the embedding vector.')
    # used in batch extraction
    # parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size for inference images.')

    parser.add_argument('-cc', '--container_capacity', type=int, default=165000,
                        help='Maximum capacity of internal data buffer.')

    args = parser.parse_args()
    main(args)
