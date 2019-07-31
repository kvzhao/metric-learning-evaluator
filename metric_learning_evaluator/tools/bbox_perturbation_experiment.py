"""Bounding Box Perturbation Experiment
  Input: DatasetBackbone with checkout images (large image contains several instances)
  Output: Saved EmbeddingContainer (with attributes denote operations)
"""

import yaml
import numpy as np

from tqdm import tqdm
from scutils.scdata import DatasetBackbone

from metric_learning_evaluator.inference.utils.image_utils import crop_and_resize
from metric_learning_evaluator.inference.utils.image_utils import read_jpeg_image
from metric_learning_evaluator.inference.utils.image_utils import shift_center_by_offset
from metric_learning_evaluator.inference.utils.image_utils import enlarge_box_by_offset
from metric_learning_evaluator.inference.utils.image_utils import shrink_box_by_offset
from metric_learning_evaluator.inference.components.extractor import FeatureExtractor
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer

# in pixels
SIZE_PERTURBATION = 10
SHIFT_PERTURBATION = 15

CASES = [
    'origin',
    'shift_left',
    'shift_right',
    'shift_up',
    'shift_down',
    'shift_upper_left',
    'shift_upper_right',
    'shift_lower_left',
    'shift_lower_right',
    #'enlarge',
    #'shrink',
    #'shift_left_enlarge',
]


def main(args):
    data_dir = args.data_dir
    out_dir = args.out_dir
    config_path = args.config_path
    sample_n = args.sample_n
    with open(config_path, 'r') as fp:
        config_dict = yaml.load(fp)
    extractor_setting = config_dict['extractor_settings']
    image_size = extractor_setting['image_size']
    model_path = extractor_setting['model_path']
    embedding_size = extractor_setting['embedding_size']

    src_db = DatasetBackbone(data_dir)
    filenames = src_db.query_all_img_filenames()

    if sample_n is not None:
        filenames = sorted(filenames)[: sample_n]
    print('#of images: {}'.format(len(filenames)))

    feature_extractor = FeatureExtractor(pb_model_path=model_path,
                                         resize=image_size)

    container = EmbeddingContainer(embedding_size,
                                   probability_size=0,
                                   container_size=args.container_size,
                                   name='perturbation')

    instance_counter = 0
    for filename in tqdm(filenames):
        annotations = src_db.query_anno_info_by_filename(filename)
        image_file_path = src_db.query_img_abspath_by_filename(filename)[0]

        image = read_jpeg_image(image_file_path)
        for annotation in annotations:
            image_id = int(annotation['img_id'])
            annotation_id = int(annotation['id'])
            label_id = int(annotation['cate_id'])
            label_name = annotation['category']
            original_bbox = annotation['bbox']

            perturbed_features = {}
            """
              origin, shift, size
              add container with attributes
            """
            for case in CASES:
                if case == 'origin':
                    bbox = original_bbox
                if case == 'shift_up':
                    bbox = shift_center_by_offset(original_bbox, offset_y=SHIFT_PERTURBATION)
                if case == 'shift_down':
                    bbox = shift_center_by_offset(original_bbox, offset_y=-SHIFT_PERTURBATION)
                if case == 'shift_right':
                    bbox = shift_center_by_offset(original_bbox, offset_x=SHIFT_PERTURBATION)
                if case == 'shift_left':
                    bbox = shift_center_by_offset(original_bbox, offset_x=-SHIFT_PERTURBATION)
                if case == 'shift_upper_left':
                    bbox = shift_center_by_offset(original_bbox,
                                                  offset_x=-SHIFT_PERTURBATION,
                                                  offset_y=SHIFT_PERTURBATION)
                if case == 'shift_upper_right':
                    bbox = shift_center_by_offset(original_bbox,
                                                  offset_x=SHIFT_PERTURBATION,
                                                  offset_y=SHIFT_PERTURBATION)
                if case == 'shift_lower_right':
                    bbox = shift_center_by_offset(original_bbox,
                                                  offset_x=SHIFT_PERTURBATION,
                                                  offset_y=-SHIFT_PERTURBATION)
                if case == 'shift_lower_left':
                    bbox = shift_center_by_offset(original_bbox,
                                                  offset_x=-SHIFT_PERTURBATION,
                                                  offset_y=-SHIFT_PERTURBATION)
                if case == 'enlarge':
                    bbox = enlarge_box_by_offset(original_bbox, SIZE_PERTURBATION)
                if case == 'shrink':
                    bbox = shrink_box_by_offset(original_bbox, SIZE_PERTURBATION)

                img = crop_and_resize(image, bbox, image_size)
                feat = feature_extractor.extract_feature(img)
                perturbed_features[case] = feat

            for case, feature in perturbed_features.items():
                container.add(instance_id=instance_counter,
                              label_id=label_id,
                              label_name=label_name,
                              filename=image_file_path,
                              embedding=feature,
                              attributes={
                                  'perturb_case': case,
                                  'annotation_id': annotation_id,
                                  'image_id': image_id,
                              })
                instance_counter += 1

    container.save(out_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Experiment: Influence of bounding box fluctuation toward embeddings.')
    parser.add_argument('-c', '--config_path', type=str, default=None)
    parser.add_argument('-dd', '--data_dir', type=str, default=None,
                        help='')
    parser.add_argument('-od', '--out_dir', type=str, default=None,
                        help='')
    parser.add_argument('--sample_n', type=int, default=None,
                        help='')
    parser.add_argument('--container_size', type=int, default=100000,
                        help='')
    args = parser.parse_args()
    main(args)
