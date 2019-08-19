# **inference**

## components

Use [Cradle](http://awsgit.viscovery.co/Cybertron/Cradle/) to replace components

Only need to focus on pipeline logic and applications.

## pipeline

- two stages
- single stage

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

## Notes

Can we make this folder stand-alone?
