# Metric Learning Evaluator

## Introduction

The repo provides an metric learning evaluation tool which support online (training time) evaluation and offline performance & report generation. Several features the evaluator provides
1. Modularization of each performance measures
2. Efficient and reusable data container
3. Several command line tools for smooth data manipulation

TODO: This version is targeted for 1.0.0 publish.
1. Ready for deployment
2. Online evaluation: more detailed measures (e.g. purity, margin..)
3. Offline evaluation: complete report generation
4. DB-verification supports


## Installation

Install **metric_learning_evaluator**

```
python setup.py install
```
Two command-line tools will be installed
- `ml-evaluation`
- `ml-inference`

### Dependencies and `hnswlib`
Run installation script
```bash
  sh install_hnsw.sh
```

## Usage

Extract features
```
ml-inference -c configs/ -dd <path_to_dataset_backbone> -od <path_to_feature_object>
```

Evaluate performance

```
ml-evaluation -c configs/eval_config.yml -dd <path_to_feature_object>
```


### Applications
**Extract features and save as EmbeddingContainier**
Two kinds of input are supported
- folder
  - structure folder, for example
  ```bash
        20190815_repeated/
        |-- 45141164
        |   |-- 20190810-123145_877.JPG
        |   |-- 20190810-123149_617.JPG
        |   |-- 20190810-123153_477.JPG
        |   |-- 20190810-123157_403.JPG
        |   `-- 20190810-123201_115.JPG
        |-- 45164804
        |   |-- 20190810-123406_888.JPG
        |   |-- 20190810-123411_307.JPG
        |   |-- 20190810-123414_922.JPG
        |   |-- 20190810-123419_248.JPG
        |   `-- 20190810-123422_616.JPG
        |-- 45173639
        |   |-- 20190810-124026_046.JPG
        |   |-- 20190810-124030_040.JPG
        |   |-- 20190810-124033_827.JPG
        |   |-- 20190810-124037_939.JPG
        |   `-- 20190810-124041_653.JPG
  ```
- csv
  - column name convention
  ```
  image_path,instance_id,label_id,label_name,type,seen_or_unseen,source
  /vol/08812401/kevin.zhao/SRData/Standard/merged/SRMega/images/00002515.jpg,2515,1,An-donuts,bread,unseen,Fuko
  /vol/08812401/kevin.zhao/SRData/Standard/merged/SRMega/images/00003441.jpg,3441,1,An-donuts,bread,unseen,Fuko
  /vol/08812401/kevin.zhao/SRData/Standard/merged/SRMega/images/00000085.jpg,85,1,An-donuts,bread,unseen,Fuko
  /vol/08812401/kevin.zhao/SRData/Standard/merged/SRMega/images/00000056.jpg,56,1,An-donuts,bread,unseen,Fuko
  ```

Execution command
```bash
  ml-inference -c extract_config.yaml \
               -t extract \
               -dd <image_folder or csv_path> \
               -dt folder \
               -od <output_folder_path>
```
- dt (data_type) can be `csv` or `folder`


Source: [hnswlib](https://github.com/nmslib/hnswlib)
Binding installation
```
git clone https://github.com/nmslib/hnswlib
apt-get install -y python-setuptools python-pip
pip3 install pybind11 numpy setuptools
cd hnswlib/python_bindings
python3 setup.py install
```

## Usage
How to use evaluator?
- Online mode

On-line evaluation is embedded in `tf-metric-learning` repo, for more detailed please refer to `tf-metric-learning/builders/evaluator_builder.py`.

Estimator will execute evaluations and provide info like
```
INFO:tensorflow:Saving dict for global step 1500: global_step = 1500, loss = 7.452976, rank-all_classes-mAP = 0.495, rank-all_classes-top_k_hit_accuracy-@k=1 = 0.49666667, rank-all_classes-top_k_hit_accuracy-@k=5 = 0.49666667
```

- Off-line mode

NOTE: There will be a new usage logic and operations  .

One can use command-line tool called `ml-eval` to execute evaluations.

```
usage: Command-line Metric Learning Evaluation Tool [-h] [--config CONFIG]
                                                    [--data_dir DATA_DIR]
                                                    [--data_type DATA_TYPE]
                                                    [--out_dir OUT_DIR]
                                                    [--embedding_size EMBEDDING_SIZE]
                                                    [--logit_size LOGIT_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Path to the evaluation configuration with yaml format.
  --data_dir DATA_DIR, -dd DATA_DIR
                        Path to the source dataset, tfrecord |
                        dataset_backbone | folder
  --data_type DATA_TYPE, -dt DATA_TYPE
                        Type of the input dataset.
  --out_dir OUT_DIR, -od OUT_DIR
                        Path to the output dir for saving report.
  --embedding_size EMBEDDING_SIZE, -es EMBEDDING_SIZE
                        Dimension of the given embeddings.
  --logit_size LOGIT_SIZE, -ls LOGIT_SIZE
                        Size of the logit used in container.
```

for example

```
ml_evaluator -c eval_config.yml -dd extracted_embeddings_facenet-centerloss-batch512
```

NOTE: Off-line mode not fully supported now.


## System Overview

[Slide: Introduction to metric learning evaluator](https://docs.google.com/presentation/d/1kSiPbLofAJ1W46IV0TKONhhGPCtsuis3RWezKKR88x8/edit?usp=sharing)

![](figures/tf-metric-evaluator_v0.3.png)


### Modules & Components
- `application`: Command-line applications
- `analysis`: Analysis tools
- `core`: Define standard fields
- config_parser
- `evaluations`: Customized applications for measuring model performance
- `metrics`: Computational objects used in evaluations
- `index`: Provide fast algorithm for query features and distance functions
- `query`: Attribute database general interface
- `tools`: Scripts for some utilities
  - NOTE: should we promote to analysis tool?
- `data_tools`: General data containers including embedding, attribute and result containers
- `utils`: Contains sampler, switcher
- `inference`: Tools for extracting features, detect boxes and pre-labeling, which can be used calling `ml-inference`.

### Supported Data Format

#### Label map
Save in json, with two layers structure (dict of dict)
```
  {
    "": {
    }
  }
```

#### Attribute table
Save in csv format, with 4 must have column names:
```
```

#### Data CSV

#### EmbeddingContainer & EmbeddingDB
- EmbeddingContainer: The container is fully supported in evaluation system
- EmbeddingDB: `metric_learning_evaluator` provides seamless conversion methods to Cradle EmbeddingDB if it follows several convention

##### EmbeddingContainer

##### EmbeddingDB
Several keywords should be provided in `meta_dict`
```python
meta_dict = {
  'instance_ids'  : [],
  'label_ids'     : [],
  'label_names'   : [],
}
```


### Roadmap
- inference
- analysis
- front-end gui

## ISSUES
- Pushed embeddings are more than container size.
- Configuration and metric names are not standard
- attribute container should be the only way that external information is added
  - we should parse grouping_rules.json in parser
- number of sampled instances


## Cooperative Repo
- tf-metric-learning
