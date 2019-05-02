"""
Two stage inference
    - detector
    - classifier

This will be integrated into `cmdline`
"""

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

from os import listdir
from os.path import join, isfile

import cv2
import numpy as np

import yaml

import scutils

from metric_learning_evaluator.inference.app.retrieval import ImageRetrieval

from metric_learning_evaluator.data_tools.feature_object import FeatureObject

from metric_learning_evaluator.inference.utils.image_utils import read_jpeg_image

from metric_learning_evaluator.inference.utils.visualization_tools import vis_one_image

from metric_learning_evaluator.data_tools.image_object import ImageObjectStandardFields as img_fields


"""TODO:
    - Support prelabeling?
    - Cropbox
    - Video streaming as input
    - How to support different input pipe?
    - How to update features?
    - Can we modularize re-ranking mechanism
    - Can I build up the serving and front-end like a demo system?
    - Should we merge all three major options?
"""

def main(args):
    
    config_path = args.config_path
    data_dir = args.data_dir
    out_dir = args.out_dir

    try:
        with open(config_path, "r") as fp:
            configs = yaml.load(fp)
    except:
        raise ValueError('Config Path: {} can not be loaded.'.format(config_path))


    with_groundtruth = False
    if os.path.exists(join(data_dir, 'dataset_backbone.db')):
        # If the input is datasetbackbone
        # Should Not be necessary
        from scutils.scdata import DatasetBackbone

        with_groundtruth = True
        src_db = DatasetBackbone(data_dir)
        filenames = src_db.query_all_img_filenames()
        input_filenames = src_db.query_img_abspath_by_filename(filenames)
        print('Load input data from datasetbackbone with {} images.'.format(len(input_filenames)))
    else:
        input_filenames = [
            join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
        print('Load input data from folder with {} images.'.format(len(input_filenames)))

    retrieval_module = ImageRetrieval(configs)

    """
      There is no need to generate dataset backbone
      We can dump annotation only. --> improve image_object
    """
    if out_dir is None:
        out_dir = 'tmp/output_two_stage_inference'
        print('Notice: out_dir is not give, save to {} in default.'.format(out_dir))
    dst_db = scutils.scdata.DatasetBackbone(out_dir)

    # NOTE: Suppose the input is dataset backbone.
    predicted_image_objects = []
    for fname in input_filenames:
        image = read_jpeg_image(fname)

        img_obj = retrieval_module.inference(image)

        if with_groundtruth:
            groundtruth_names = [anno['category']
                for anno in src_db.query_anno_info_by_filename(os.path.basename(fname))]
            print('Groundtruth: {}'.format(', '.join(groundtruth_names)))
        # show predicted results on command-line
        print(img_obj)

        origin_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_id = dst_db.imwrite(origin_img)

        for _inst_id, _instance in img_obj.instances.items():
            _inst_label_name = _instance[img_fields.instance_label_name]
            _inst_label_id = _instance[img_fields.instance_label_id]
            _inst_bbox = _instance[img_fields.bounding_box]

            _anno_id = dst_db.init_annotation(img_id)
            dst_db.update_category(_anno_id, _inst_label_name)
            dst_db.update_bbox(_anno_id, _inst_bbox)

        predicted_image_objects.append(img_obj)

    ## Export all results
    dst_db.commit()
    embeddings = []
    filename_strings = []
    label_names, label_ids = [], []
    for _img_obj in predicted_image_objects:
        for _inst_id, _instance in _img_obj.instances.items():
            filename_strings.append(_inst_id)
            embeddings.append(_instance[img_fields.instance_feature])
            label_ids.append(_instance[img_fields.instance_label_id])
            label_names.append(_instance[img_fields.instance_label_name])

    feature_exporter = FeatureObject()
    feature_exporter.embeddings = np.asarray(embeddings)
    feature_exporter.filename_strings = np.asarray(filename_strings)
    feature_exporter.label_ids = np.asarray(label_ids)
    feature_exporter.label_names = np.asarray(label_names)
    feature_exporter.save(out_dir + '/retrieved_embeddings')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Two Stage Inference Tool')

    parser.add_argument('-c', '--config_path', type=str, default='two_stage_config.yml',
                        help='Path to the configuration yaml file.')
    parser.add_argument('-dd', '--data_dir', type=str, default=None,
                        help='Path to Input DatasetBackbone or raw image folder.')
    parser.add_argument('-od', '--out_dir', type=str, default=None,
                        help='Path to Output DatasetBackbone.')

    args = parser.parse_args()
    main(args)