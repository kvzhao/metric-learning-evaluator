# Metric Learning Evaluator

## System Architecture Overview

![](figures/tf-metric-evaluator_v0.2.png)

## Installation

```
python setup.py install
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

One can use command-line tool called `ml_evaluate` to execute evaluations.

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


## Cooperative Repo
- tf-metric-learning
