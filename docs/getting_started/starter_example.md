# Starter Tutorial

```{tip}
Make sure you've followed the [installation](installation.md) steps first.
```
Here is a starter example for using LlamaIndex. 

### Download

LlamaIndex examples can be found in the `examples` folder of the LlamaIndex repository.
We first want to download this `examples` folder. An easy way to do this is to just clone the repo:

```bash
$ git clone https://github.com/jerryjliu/llama_index.git
```

Next, navigate to your newly-cloned repository, and verify the contents:

```bash
$ cd llama_index
$ ls
LICENSE                data_requirements.txt  tests/
MANIFEST.in            examples/              pyproject.toml
Makefile               experimental/          requirements.txt
README.md              llama_index/             setup.py
```

We now want to navigate to the following folder:

```bash
$ cd examples/paul_graham_essay
```

This contains LlamaIndex examples around Paul Graham's essay, ["What I Worked On"](http://paulgraham.com/worked.html). A comprehensive set of examples are already provided in `TestEssay.ipynb`. For the purposes of this tutorial, we can focus on a simple example of getting LlamaIndex up and running.

### Build and Query Index

Create a new `.py` file with the following:

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)
```

This builds an index over the documents in the `data` folder (which in this case just consists of the essay text). We then run the following

```python
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
```

You should get back a response similar to the following: `The author wrote short stories and tried to program on an IBM 1401.`

### Viewing Queries and Events Using Logging

In a Jupyter notebook, you can view info and/or debugging logging using the following snippet:

```python
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
```

You can set the level to `DEBUG` for verbose output, or use `level=logging.INFO` for less.

### Saving and Loading

By default, data is stored in-memory.
To persist to disk (under `./storage`):

```python
index.storage_context.persist()
```

To reload from disk:
```python
from llama_index import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage")
# load index
index = load_index_from_storage(storage_context)
```



```{admonition} Next Steps
* learn more about the [high-level concepts](/getting_started/concepts.md).
* tell me how to [customize things](/getting_started/customization.rst).
* curious about a specific module? check out the guides 👈
* have a use case in mind? check out the [end-to-end tutorials](/end_to_end_tutorials/use_cases.md)
```

