****# Metric Learning Evaluator

## System Architecture Overview

![](figures/tf-metric-evaluator_v0.2.png)

### Description
- EvaluationBuilder
- EmbeddingContainer
- AttributeContainer
- QueryIterface
- Metrics

## TODOs

- ~~basic class~~
- metric functions
- config writer
- How to execute offline evaluation without embedder?
- Define the standard format of evaluation results.

## Configuration

Usage:
     The configuration is used to change hyper-parameters but not change metric items.

There are 5 required items should be defined in the configuration.
- database
- evaluation
- container_size
- embedding_size
- logit_size


### `evaluation`

Define type of metrics and attribute in each evaluations, like:

```python
classification:
    Top_k:
        - 5
    Attr:
        - Color
        - Shape
```

The format of `per_eval_config` is `dict` of `list`:

```python

evaluation_name:
    metric_type:
        - value_0
        - value_1
    Attr:
        - attribute_0
        - attribute_1
```


## Evaluated Results

```python

eval_results = {
    # e.g. classification
    'evaluation_name':
    {
        # level of attributes
        'attribute_type':
        {
            # e.g. top_k accuracy
            # NOTE: maybe this layer would be cancelled .
            'metric_type':
            {
                # e.g. IoU, top k
                'threshold': value # or list of values?
            },
        }
    }
}

```

### Example

### Report Writer

## Customized Evaluation

Steps:

- Standard Fields
  - Evaluation Standard Fields
  - Metric Standard Fiedls
  - Attribute Standard Fields
- Registration


## Cooperative Repo
- tf-metric-learning
