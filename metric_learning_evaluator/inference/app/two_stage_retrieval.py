"""
Two stage retrieval

Input
    - Image 
        * in flat folder
        * dataset backbone?
Output
    - DatasetBackbone
    - images with box & class annotation on it?
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

# DatasetBackbone dependent
import scutils
from scutils.scdata import DatasetBackbone

from metric_learning_evaluator.inference.pipeline.retrieval import ImageRetrieval

from metric_learning_evaluator.data_tools.feature_object import FeatureObject
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer

from metric_learning_evaluator.inference.utils.image_utils import read_jpeg_image
from metric_learning_evaluator.inference.utils.visualization_tools import vis_one_image
from metric_learning_evaluator.inference.utils.image_utils import load_file_from_folder
from metric_learning_evaluator.inference.utils.image_utils import load_file_from_structure_folder

from metric_learning_evaluator.core.standard_fields import ImageObjectStandardFields as img_fields
from metric_learning_evaluator.core.standard_fields import ConfigStandardFields as config_fields


def retrieval_application(configs, args):
    # parse arguments
    data_dir = args.data_dir
    out_dir = args.out_dir
    data_type = args.data_type
    label_map_path = configs.labelmap_path
    has_labelmap = False

    if label_map_path is not None:
        has_labelmap = True
        # Load label map

    with_groundtruth = False
    from_dataset_backbone = False

    input_filenames = None
    label_names = None
    label_ids = None

    if data_type == 'datasetbackbone':
        with_groundtruth = True
        from_dataset_backbone = True
        src_db = DatasetBackbone(data_dir)
        filenames = src_db.query_all_img_filenames()
        input_filenames = src_db.query_img_abspath_by_filename(filenames)
        label_names = [anno['category'] for anno in src_db.query_anno_info_by_filename(os.path.basename(input_filenames))]
        print(label_names)
        print(len(label_names), len(input_filenames))
        print('Load input data from datasetbackbone with {} images.'.format(len(input_filenames)))
    elif data_type == 'folder':
        # TODO: Use folder name as label name?
        input_filenames = load_file_from_folder(data_dir)
    elif data_type == 'nested':
        # TODO: Use folder name as label name?
        with_groundtruth = True
        input_filenames, label_names = load_file_from_structure_folder(data_dir)
        print(len(label_names), len(input_filenames))
    else:
        raise ValueError('data_type:{} not supported.'.format(data_type))

    if with_groundtruth and label_ids is None:
        # create labelmap and assign id for each
        if from_dataset_backbone:
            pass
        elif has_labelmap:
            pass
        else:
            label_name_set = set(label_names)
            labelmap = {name: index
                        for index, name in enumerate(label_name_set)}
            label_ids = [labelmap[name] for name in label_names]

    retrieval_module = ImageRetrieval(configs)

    """
      There is no need to generate dataset backbone
      We can dump annotation only. --> improve image_object
    """
    if out_dir is None:
        out_dir = 'tmp/output_two_stage_inference'
        print('Notice: out_dir is not give, save to {} in default.'.format(out_dir))

    # TODO: due to box, we must save them as dataset backbone
    dst_db = scutils.scdata.DatasetBackbone(out_dir)

    # TODO: Remove this
    predicted_image_objects = []

    capacity = configs[config_fields.embedding_container_capacity]
    emb_size = configs[config_fields.extractor_settings][config_fields.embedding_size]

    # TODO: Need more flexible container size
    container = EmbeddingContainer(embedding_size=emb_size,
                                   container_size=capacity,
                                   name='two_stage_result')
    for idx, filename in enumerate(input_filenames):
        try:
            image = read_jpeg_image(filename)
        except:
            print('{} can not be opened properly, skip.'.format(filename))
            continue

        img_obj = retrieval_module.inference(image)

        # show predicted results on command-line
        print(img_obj)

        origin_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_id = dst_db.imwrite(origin_img)

        for _inst_id, _instance in img_obj.instances.items():

            if with_groundtruth:
                if label_names is not None:
                    origin_label_name = _instance[img_fields.instance_label_name]
                    _instance[img_fields.instance_label_name] = label_names[idx]
                    print('update name {} -> {}'.format(origin_label_name, label_names[idx]))
                if label_ids is not None:
                    origin_label_id = _instance[img_fields.instance_label_id]
                    _instance[img_fields.instance_label_id] = label_ids[idx]
                    print('update id {} -> {}'.format(origin_label_id, label_ids[idx]))

            _inst_label_name = _instance[img_fields.instance_label_name]
            _inst_label_id = _instance[img_fields.instance_label_id]
            _inst_bbox = _instance[img_fields.bounding_box]

            _anno_id = dst_db.init_annotation(img_id)

            dst_db.update_category(_anno_id, _inst_label_name)
            dst_db.update_bbox(_anno_id, _inst_bbox)

            container.add(instance_id=_inst_id,
                          embedding=_instance[img_fields.instance_feature],
                          label_id=_inst_label_id,
                          label_name=_inst_label_name,
                          filename=filename,
                          attribute={
                              'anno_id': _anno_id,
                          })

        predicted_image_objects.append(img_obj)

    # Export all results
    dst_db.commit()
    print('Save predicted results in DatasetBackbone to {}'.format(out_dir))
    container.save(out_dir + '/extracted_features')
    print('Save extracted embeddings to {}'.format(out_dir + '/extracted_features'))
    print(container)
