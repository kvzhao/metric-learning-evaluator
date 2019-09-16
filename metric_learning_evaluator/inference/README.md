# **inference**

## components

(TODO) Use [Cradle](http://awsgit.viscovery.co/Cybertron/Cradle/) to replace components
Only need to focus on pipeline logic and applications.

## pipeline

- two stages
- single stage
  - detection crop
  - feature extraction

## app

applications like

1) extractor feature and save as anchor database
2) bounding box prelabel
3) serve a feature database

tools:
- utils

## scripts
Some useful tools for import/export features or pre-labelling

- `sequential_feature_extraction.py`

The reliable way to extract features is using the following script.
```
    metric_learning_evaluator/tools/feature_extraction_from_csv.py
```

### Following contents are some notes for inference model type

Can we make this folder stand-alone? And move tfmodel & tflite to components

## TFLite Inference Code

### TFLite Model Conversion

#### Convert SavedModel (Extractor)

Command line tool
```bash
tflite_convert \
  --output_file=/tmp/foo.tflite \
  --saved_model_dir=/tmp/saved_model
```

Python script
```bash
python lite_converter.py
  --output_file=/tmp/foo.tflite \
  --saved_model_dir=/tmp/saved_model
```

#### Convert GraphDef
```bash
tflite_convert \
  --output_file=/tmp/foo.tflite \
  --graph_def_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --input_arrays=input \
  --output_arrays=MobilenetV1/Predictions/Reshape_1
```

If not sure, `python print_tensor_graphdef.py -pb <path_to_pb>` can help

#### Convert GraphDef (Detector)

For **detector**, people should use [`export_tflite_ssd_graph.py`](https://github.com/tensorflow/models/blob/master/research/object_detection/export_tflite_ssd_graph.py) converting checkpoint to graph.pb, then run the procedure. (Which defines input shape & switch operators)

1. Export GraphDef

```bash
python3 export_tflite_ssd_graph.py \ --pipeline_config_path=/notebooks/git/models/research/object_detection/ssd_training/kv_ssd_inception/pipeline.config \ --trained_checkpoint_prefix=/notebooks/git/models/research/object_detection/ssd_training/kv_ssd_inception/model.ckpt-50000 \ --output_directory=/notebooks/git/models/research/object_detection/ssd_training/kv_ssd_inception/freeze/ssd_inception_for_tflite.pb \ --add_postprocessing_op=true
```

2. Convert .tflite

```bash
tflite_convert \
--output_file=kv_product_inception_agnostic.tflite \ 
--graph_def_file=kv_product_inception_agnostic.pb \ 
--input_arrays=normalized_input_image_tensor \
--input_shapes=1,300,300,3 \ 
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2' 'TFLite_Detection_PostProcess:3' \
--mean_values=128 \ 
--std_dev_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops
```

NOTE: the convertion config is referred to [github issue tensorflow/models:#4838](https://github.com/tensorflow/models/issues/4838#issuecomment-406341521)

### Inference Interface

TODO: @kv Move these codes to Cradle.

#### Detector
```python
detector = TfliteDetector(model_path=model_path)

raw_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2BGR)

raw_image.resize(300, 300, 3)

start = time.time()
results = detector.detect(raw_image)
print('dT = {}'.format(1000.0 * (time.time() - start)))

# SSD Inception_v2
# dT = 641.0093307495117  (warmup)
# dT = 191.82753562927246 (inference)
```

#### Feature Extractor

```python
feature_extractor = TfliteExtractor(model_path=model_path)
raw_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2BGR)

start = time.time()
feature_extractor.extract(raw_image)
print('dT = {}'.format(1000.0 * (time.time() - start)))

# MobileNet_v2
# dT = 178.9078712463379 (warmup)
# dT = 67.14677810668945 (inference)
```


## GRPC Serving