


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
from metric_learning_evaluator.inference.components.detector_base import DetectorBase
from metric_learning_evaluator.inference.components.detector_base import DetectorStandardFields
detector_fields = DetectorStandardFields

class Detector(DetectorBase):

    def __init__(self, pb_model_path, labelmap_path, num_classes):
        super(Detector, self).__init__(pb_model_path, labelmap_path, num_classes)

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
                    detector_fields.num_detections,
                    detector_fields.detection_classes,
                    detector_fields.detection_scores,
                    detector_fields.detection_boxes,
                    ]:
                        tensor_name = key + ':0'
                        if tensor_name in all_tensor_names:
                            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                self.sess = sess
                self.image_tensor = image_tensor
                self.tensor_dict = tensor_dict
        print ("Detector initialized.")

    def _run_inference(self, raw_image, threshold=0.5):
        detections = {}
        output_dict = self.sess.run(
            self.tensor_dict, feed_dict = {self.image_tensor: np.expand_dims(raw_image, 0)})
        output_dict[detector_fields.detection_scores] = output_dict[detector_fields.detection_scores][0]
        valid_idxs = np.where(output_dict[detector_fields.detection_scores] > threshold)[0]

        detections[detector_fields.detection_classes] = output_dict[
            detector_fields.detection_classes][0].astype(np.uint8)[valid_idxs]
        detections[detector_fields.detection_boxes] = output_dict[
            detector_fields.detection_boxes][0][valid_idxs]
        detections[detector_fields.detection_scores] = output_dict[
            detector_fields.detection_scores][valid_idxs]

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