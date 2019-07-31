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
from metric_learning_evaluator.inference.utils.image_utils import bbox_center_offset
from metric_learning_evaluator.inference.utils.image_utils import bbox_size_offset
from metric_learning_evaluator.inference.components.extractor import FeatureExtractor
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer


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
    'enlarge',
    'shrink',
    'shift_left_enlarge',
    'shift_right_enlarge',
    'shift_up_enlarge',
    'shift_down_enlarge',
    'shift_left_shrink',
    'shift_right_shrink',
    'shift_up_shrink',
    'shift_down_shrink',
    'shift_upper_left_enlarge',
    'shift_upper_right_enlarge',
    'shift_lower_left_enlarge',
    'shift_lower_right_enlarge',
    'shift_upper_left_shrink',
    'shift_upper_right_shrink',
    'shift_lower_left_shrink',
    'shift_lower_right_shrink',
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

    # in pixels
    translation_perturbation = args.translation_shift
    size_perturbation = args.size_shift

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
            for case in CASES:
                if case == 'origin':
                    bbox = original_bbox
                if case == 'shift_up':
                    bbox = bbox_center_offset(original_bbox, offset_y=translation_perturbation)
                if case == 'shift_down':
                    bbox = bbox_center_offset(original_bbox, offset_y=-translation_perturbation)
                if case == 'shift_right':
                    bbox = bbox_center_offset(original_bbox, offset_x=translation_perturbation)
                if case == 'shift_left':
                    bbox = bbox_center_offset(original_bbox, offset_x=-translation_perturbation)
                if case == 'shift_upper_left':
                    bbox = bbox_center_offset(original_bbox,
                                              offset_x=-translation_perturbation,
                                              offset_y=translation_perturbation)
                if case == 'shift_upper_right':
                    bbox = bbox_center_offset(original_bbox,
                                              offset_x=translation_perturbation,
                                              offset_y=translation_perturbation)
                if case == 'shift_lower_right':
                    bbox = bbox_center_offset(original_bbox,
                                              offset_x=translation_perturbation,
                                              offset_y=-translation_perturbation)
                if case == 'shift_lower_left':
                    bbox = bbox_center_offset(original_bbox,
                                              offset_x=-translation_perturbation,
                                              offset_y=-translation_perturbation)
                if case == 'enlarge':
                    bbox = bbox_size_offset(original_bbox, size_perturbation)
                if case == 'shrink':
                    bbox = bbox_size_offset(original_bbox, -size_perturbation)
                if case == 'shift_left_enlarge':
                    bbox = bbox_center_offset(original_bbox, offset_x=-translation_perturbation)
                    bbox = bbox_size_offset(bbox, size_perturbation)
                if case == 'shift_right_enlarge':
                    bbox = bbox_center_offset(original_bbox, offset_x=translation_perturbation)
                    bbox = bbox_size_offset(bbox, size_perturbation)
                if case == 'shift_up_enlarge':
                    bbox = bbox_center_offset(original_bbox, offset_y=translation_perturbation)
                    bbox = bbox_size_offset(bbox, size_perturbation)
                if case == 'shift_down_enlarge':
                    bbox = bbox_center_offset(original_bbox, offset_y=-translation_perturbation)
                    bbox = bbox_size_offset(bbox, size_perturbation)
                if case == 'shift_left_shrink':
                    bbox = bbox_center_offset(original_bbox, offset_x=-translation_perturbation)
                    bbox = bbox_size_offset(bbox, -size_perturbation)
                if case == 'shift_right_shrink':
                    bbox = bbox_center_offset(original_bbox, offset_x=translation_perturbation)
                    bbox = bbox_size_offset(bbox, -size_perturbation)
                if case == 'shift_up_shrink':
                    bbox = bbox_center_offset(original_bbox, offset_y=translation_perturbation)
                    bbox = bbox_size_offset(bbox, -size_perturbation)
                if case == 'shift_down_shrink':
                    bbox = bbox_center_offset(original_bbox, offset_y=-translation_perturbation)
                    bbox = bbox_size_offset(bbox, -size_perturbation)
                if case == 'shift_upper_left_enlarge':
                    bbox = bbox_center_offset(original_bbox,
                                              offset_x=-translation_perturbation,
                                              offset_y=translation_perturbation)
                    bbox = bbox_size_offset(bbox, size_perturbation)
                if case == 'shift_upper_right_enlarge':
                    bbox = bbox_center_offset(original_bbox,
                                              offset_x=translation_perturbation,
                                              offset_y=translation_perturbation)
                    bbox = bbox_size_offset(bbox, size_perturbation)
                if case == 'shift_lower_right_enlarge':
                    bbox = bbox_center_offset(original_bbox,
                                              offset_x=translation_perturbation,
                                              offset_y=-translation_perturbation)
                    bbox = bbox_size_offset(bbox, size_perturbation)
                if case == 'shift_lower_left_enlarge':
                    bbox = bbox_center_offset(original_bbox,
                                              offset_x=-translation_perturbation,
                                              offset_y=-translation_perturbation)
                    bbox = bbox_size_offset(bbox, size_perturbation)
                if case == 'shift_upper_left_shrink':
                    bbox = bbox_center_offset(original_bbox,
                                              offset_x=-translation_perturbation,
                                              offset_y=translation_perturbation)
                    bbox = bbox_size_offset(bbox, -size_perturbation)
                if case == 'shift_upper_right_shrink':
                    bbox = bbox_center_offset(original_bbox,
                                              offset_x=translation_perturbation,
                                              offset_y=translation_perturbation)
                    bbox = bbox_size_offset(bbox, -size_perturbation)
                if case == 'shift_lower_right_shrink':
                    bbox = bbox_center_offset(original_bbox,
                                              offset_x=translation_perturbation,
                                              offset_y=-translation_perturbation)
                    bbox = bbox_size_offset(bbox, -size_perturbation)
                if case == 'shift_lower_left_shrink':
                    bbox = bbox_center_offset(original_bbox,
                                              offset_x=-translation_perturbation,
                                              offset_y=-translation_perturbation)
                    bbox = bbox_size_offset(bbox, -size_perturbation)

                if not bbox:
                    print('img:{} id:{} op={} has empty bbox, skip'.format(image_id, annotation_id, case))
                    continue
                try:
                    img = crop_and_resize(image, bbox, image_size)
                except:
                    print('img:{} id:{} op={} crop bbox={} fails!, skip'.format(image_id, annotation_id, case, bbox))
                    continue

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
    parser.add_argument('-ts', '--translation_shift', type=int, default=15,
                        help='')
    parser.add_argument('-ss', '--size_shift', type=int, default=15,
                        help='')
    args = parser.parse_args()
    main(args)
