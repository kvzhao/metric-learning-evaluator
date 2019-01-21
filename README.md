# Metric Learning Evaluator

## Overview

![](figures/tf-metric-evaluator_v0.2.png)



## TODOs

- ~~basic class~~
- metric functions
- config writer
- How to execute offline evaluation without embedder?
- Define the standard format of evaluation results.

## Evaluated Results

```python

eval_results = {
    'evaluation_name':
    {
        'metric_type':
        {
            'per_category':
            {
                'some_threshold': [values]
            },

            'per_attribute': {}, #?
            'thresholds': [], # values
        }
    }
}

```

## Cooperative Repo
- tf-metric-learning
