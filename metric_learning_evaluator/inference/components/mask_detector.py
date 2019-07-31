


import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import tensorflow as tf
import tensorflow.contrib


from metric_learning_evaluator.inference.utils import label_map_util
from metric_learning_evaluator.inference.utils.image_utils import bbox_ratio_to_xywh
from metric_learning_evaluator.inference.utils.image_utils import bbox_xywh_to_corner_format
from metric_learning_evaluator.inference.components.mask_detector_base import MaskDetectorBase
from metric_learning_evaluator.inference.components.mask_detector_base import MaskDetectorStandardFields
mask_detector_fields = MaskDetectorStandardFields

class MaskDetector(DetectorBase):

    def __init__(self, pb_model_path, labelmap_path, num_classes,
                       image_height, image_width):
        super(MaskDetector, self).__init__(pb_model_path, labelmap_path, num_classes,
                                           image_height, image_width)

        self._model_init()

    def _model_init(self):

        self._labelmap = label_map_util.load_labelmap(self._labelmap_path)
        categories = label_map_util.convert_label_map_to_categories(
                        self._labelmap,
                        max_num_classes=self._num_classes,
                        use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        self.det_graph = tf.Graph()
        with self.det_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self._pb_model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True

                sess = tf.Session(config=config)
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = { output.name for op in ops for output in op.outputs }
                tensor_dict = {}
                for key in [
                    mask_detector_fields.num_detections,
                    mask_detector_fields.detection_classes,
                    mask_detector_fields.detection_scores,
                    mask_detector_fields.detection_boxes,
                    ]:
                        tensor_name = key + ':0'
                        if tensor_name in all_tensor_names:
                            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                            tensor_dict = self._add_mask_graph(tensor_dict)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                self.sess = sess
                self.image_tensor = image_tensor
                self.tensor_dict = tensor_dict
        print ("Mask Detector initialized.")

    def _run_inference(self, raw_image, threshold=0.5):
        detections = {}
        output_dict = self.sess.run(
            self.tensor_dict, feed_dict = {self.image_tensor: np.expand_dims(raw_image, 0)})
        output_dict[mask_detector_fields.detection_scores] = output_dict[mask_detector_fields.detection_scores][0]
        valid_idxs = np.where(output_dict[mask_detector_fields.detection_scores] > threshold)[0]

        detections[mask_detector_fields.detection_classes] = output_dict[
            mask_detector_fields.detection_classes][0].astype(np.uint8)[valid_idxs]
        detections[mask_detector_fields.detection_boxes] = output_dict[
            mask_detector_fields.detection_boxes][0][valid_idxs]
        detections[mask_detector_fields.detection_scores] = output_dict[
            mask_detector_fields.detection_scores][valid_idxs]

        return detections

    def detect(self, raw_image, threshold=0.5):
        """
          Args:
            image_object: ImageObject
          Returns:
            image_object: Annotated input image_object
        """

        # NOTE @kv: Consider return instance_id
        detected = self._run_inference(raw_image, threshold)

        return detected
    
    def _add_mask_graph(self, tensor_dict):
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict[mask_detector_fields.detection_boxes], [0])
        detection_masks = tf.squeeze(tensor_dict[mask_detector_fields.detection_masks], [0])

        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict[mask_detector_fields.num_detections][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, self._image_height, self._image_widhth)
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict[mask_detector_fields.detection_masks] = tf.expand_dims(
            detection_masks_reframed, 0)
        return tensor_dict

def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width):
  """Transforms the box masks back to full image masks.

  Embeds masks in bounding boxes of larger masks whose shapes correspond to
  image shape.

  Args:
    box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
           corners. Row i contains [ymin, xmin, ymax, xmax] of the box
           corresponding to mask i. Note that the box corners are in
           normalized coordinates.
    image_height: Image height. The output mask will have the same height as
                  the image height.
    image_width: Image width. The output mask will have the same width as the
                 image width.

  Returns:
    A tf.float32 tensor of size [num_masks, image_height, image_width].
  """
  # TODO(rathodv): Make this a public function.
  def reframe_box_masks_to_image_masks_default():
    """The default function when there are more than 0 box masks."""
    def transform_boxes_relative_to_boxes(boxes, reference_boxes):
      boxes = tf.reshape(boxes, [-1, 2, 2])
      min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
      max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
      transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
      return tf.reshape(transformed_boxes, [-1, 4])

    box_masks_expanded = tf.expand_dims(box_masks, axis=3)
    num_boxes = tf.shape(box_masks_expanded)[0]
    unit_boxes = tf.concat(
        [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
    return tf.image.crop_and_resize(
        image=box_masks_expanded,
        boxes=reverse_boxes,
        box_ind=tf.range(num_boxes),
        crop_size=[image_height, image_width],
        extrapolation_value=0.0)

  image_masks = tf.cond(
      tf.shape(box_masks)[0] > 0,
      reframe_box_masks_to_image_masks_default,
      lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))
  return tf.squeeze(image_masks, axis=3)
