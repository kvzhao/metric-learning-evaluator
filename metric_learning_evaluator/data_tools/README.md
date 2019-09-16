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