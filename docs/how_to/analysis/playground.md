# Playground

The Playground module in LlamaIndex is a way to automatically test your data (i.e. documents) across a diverse combination of indices, models, embeddings, modes, etc. to decide which ones are best for your purposes. More options will continue to be added.

For each combination, you'll be able to compare the results for any query and compare the answers, latency, tokens used, and so on.

You may initialize a Playground with a list of pre-built indices, or initialize one from a list of Documents using the preset indices.

### Sample Code

A sample usage is given below.

```python
from llama_index import download_loader
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.indices.tree.base import TreeIndex
from llama_index.playground import Playground

# load data 
WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
documents = loader.load_data(pages=['Berlin'])

# define multiple index data structures (vector index, list index)
indices = [VectorStoreIndex(documents), TreeIndex(documents)]

# initialize playground
playground = Playground(indices=indices)

# playground compare
playground.compare("What is the population of Berlin?")

```

### API Reference

[API Reference here](/reference/playground.rst)


### Example Notebook

[Link to Example Notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/analysis/PlaygroundDemo.ipynb).  


```{toctree}
---
caption: Examples
maxdepth: 1
---
../../examples/analysis/PlaygroundDemo.ipynb
```