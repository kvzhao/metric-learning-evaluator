
"""
  Two stage inference
    Input:
        structure folder with raw images
    Output:
        DatasetBackbone
"""

import os
import sys
import yaml

from os import listdir
from os.path import join
from os.path import isfile

import cv2
import numpy as np

# DatasetBackbone dependent
import scutils
from scutils.scdata import DatasetBackbone

from tqdm import tqdm

# data objects
from metric_learning_evaluator.data_tools.image_object import ImageObject
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
# inference models
from metric_learning_evaluator.inference.components.detector import Detector
from metric_learning_evaluator.inference.components.extractor import FeatureExtractor
# reader & utilities
from metric_learning_evaluator.inference.utils.image_utils import read_jpeg_image
from metric_learning_evaluator.inference.utils.read_json_labelmap import read_json_labelmap
from metric_learning_evaluator.inference.utils.image_utils import bbox_ratio_to_xywh
from metric_learning_evaluator.inference.utils.image_utils import crop_and_resize
# fields
from metric_learning_evaluator.core.standard_fields import ImageObjectStandardFields
from metric_learning_evaluator.core.standard_fields import DetectorStandardFields
from metric_learning_evaluator.core.standard_fields import ImageRetrievalStandardFields

config_fields = ImageRetrievalStandardFields
img_fields = ImageObjectStandardFields
detector_fields = DetectorStandardFields


def structure_folder_retrieval(configs, args):
    """
      One layer structure
    """

    data_dir = args.data_dir
    out_dir = args.out_dir

    input_filenames = []
    for root, subdirs, files in os.walk(data_dir):
        list_file_path = os.path.join(root, 'my-directory-list.txt')

        with open(list_file_path, 'wb') as list_file:
            for subdir in subdirs:
                print('\t- subdirectory ' + subdir)

            for filename in files:
                file_path = os.path.join(root, filename)
                file_path.lower().endswith(('.png', '.jpeg', '.jpg'))
                input_filenames.append(file_path)
    print('{} contains {} images in total'.format(data_dir, len(input_filenames)))

    detector_settings = configs[config_fields.detector_settings]
    extractor_settings = configs[config_fields.extractor_settings]
    detector_num_classes = detector_settings[config_fields.num_classes]
    detector_model_path = detector_settings[config_fields.model_path]
    detector_labelmap_path = detector_settings[config_fields.labelmap_path]
    extractor_model_path = extractor_settings[config_fields.model_path]
    extractor_image_size = extractor_settings[config_fields.image_size]

    detector = Detector(pb_model_path=detector_model_path,
                        labelmap_path=detector_labelmap_path,
                        num_classes=detector_num_classes)
    
    extractor = FeatureExtractor(pb_model_path=extractor_model_path,
                                 resize=extractor_image_size)

    """
      There is no need to generate dataset backbone
      We can dump annotation only. --> improve image_object
    """

    if out_dir is None:
        out_dir = 'tmp/output_two_stage_inference'
        print('Notice: out_dir is not give, save to {} in default.'.format(out_dir))
    dst_db = scutils.scdata.DatasetBackbone(out_dir)

    # NOTE: Suppose the input is dataset backbone.

    container = EmbeddingContainer()

    label_map = {}
    label_id_counter = 0

    for fname in tqdm(input_filenames):
        try:
            image = read_jpeg_image(fname)
        except:
            print('{} can not be opened properly, skip.'.format(fname))
            continue

        origin_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        category_name_from_filename = fname.split('/')[-2]

        img_height, img_width, _ = image.shape

        detected_result = detector.detect(image)
        det_bboxes = detected_result[detector_fields.detection_boxes]
        det_scores = detected_result[detector_fields.detection_scores]

        print('{} contains {} bboxes'.format(fname, len(det_bboxes)))

        if len(det_bboxes) == 0 or args.disable_detector:
            """
              Without Detector
            """
            inst_img = cv2.resize(image, (extractor_image_size, extractor_image_size))
            inst_feat = extractor.extract_feature(inst_img)

            # push into database
            img_id = dst_db.imwrite(origin_img)
            anno_id = dst_db.init_annotation(img_id)

            if args.folder_name_as_label:
                label_name = category_name_from_filename
            else:
                label_name = 'unknown_instance'

            cate_id = dst_db.init_category(label_name)[0]
            dst_db.update_category(anno_id, label_name)

            container.add(instance_id=anno_id,
                          embedding=inst_feat,
                          label_id=cate_id,
                          filename=fname,
                          label_name=label_name)

        else:
            """
              With Detector
            """
            img_id = dst_db.imwrite(origin_img)
            for bbox, score in zip(det_bboxes, det_scores):
                formal_bbox = bbox_ratio_to_xywh(bbox, img_height, img_width)
                inst_img = crop_and_resize(image, formal_bbox, extractor_image_size)
                inst_feat = extractor.extract_feature(inst_img)
                # push into database

                if args.folder_name_as_label:
                    label_name = category_name_from_filename
                else:
                    label_name = 'unknown_instance'

                cate_id = dst_db.init_category(label_name)[0]
                anno_id = dst_db.init_annotation(img_id)
                dst_db.update_category(anno_id, label_name)
                dst_db.update_bbox(anno_id, formal_bbox)

                container.add(instance_id=anno_id,
                              embedding=inst_feat,
                              label_id=cate_id,
                              filename=fname,
                              label_name=label_name)

    print('... commit the database')
    dst_db.commit()
    print('Dataset save to {}'.format(out_dir))
    ## Export all results once.
    container.save(out_dir + '/extracted_embeddings')

    print('Done.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, default=None, help='Path to the label map.')
    parser.add_argument('-dd', '--data_dir', type=str, default=None, help='Path to data dir.')
    parser.add_argument('-od', '--out_dir', type=str, default=None, help='Path to out dir.')
    parser.add_argument('-fl', '--folder_name_as_label', action='store_true',
                        help='Use folder name as the category name if the flag is set true.')
    parser.add_argument('-ddt', '--disable_detector', action='store_true',
                        help='A flag to disable the detector inference')

    args = parser.parse_args()
    config_path = args.config_path

    # read config and pass args into the func
    try:
        with open(config_path, 'r') as fp:
            config_dict = yaml.load(fp)
    except:
        raise IOError('Can not load yaml from {}.'.format(config_path))
        # TODO: create default config instead of error.

    structure_folder_retrieval(config_dict, args)
