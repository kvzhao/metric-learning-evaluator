# Data Tools

Data tools provide several useful components
- EmbeddingContainer
- ResultContainer
- FeatureObject
- ImageObject
- AttributeTable

## *Container

In principal, container provides `add_()` function that user can push datum one by one.

### EmbeddingContainer
The embedding container is a standard (and the most important) data object within evaluator.

### Example

Path to the example `feature-examples/container_example/` which contains the following files
```bash
attribute_table.csv  embeddings.npy  filename_strings.npy  indexes.csv  instance_ids.npy  label_ids.npy  label_names.npy
```

The container pocesses the following
```python
container = EmbeddingContainer()
container.load(path)
print(container)
```
It shows 

```bash
Container:feature-examples/container_example/ created
Load embedding container from feat_obj format
feature-examples/container_example//filename_strings.npy is loaded
feature-examples/container_example//label_ids.npy is loaded
feature-examples/container_example//instance_ids.npy is loaded
feature-examples/container_example//embeddings.npy is loaded
feature-examples/container_example//label_names.npy is loaded
WARNING: Get the empty probabilities array
container size: 10000 -> 942
embedding size: 0 -> 256
Reset feature-examples/container_example/
Index Table Created
Container initialized.
=============== feature-examples/container_example/ ===============
embeddings: (942, 256)
internals: instance_ids, label_ids, label_names, filename_strings
attributes: supercategory_name, SU, supercategory_id, category_name, product_name, product_code, manufacturer_id, type, category_id, manufacturer_name
==================================================
```
The `container_example` contains
- 85 classes
  - 60 seen
  - 25 unseen
- 942 instances
  - 300 query instances
  - 642 anchor instances

It provides I/O & query interfaces

I/O
- add
- load
- save
- clear
- from_embedding_container
- from_cradle_embedding_db

Query
- get_SOMETHING_by_instance_ids


`container.save()` will dump all information on disk, which looks like
```
```


### ResultContainer


## *Object


## AttributeTable

Basically, `AttributeTable` is a pandas dataframe wrapper for evaluation purposes