# LlamaIndex Vector_Stores Integration: Hnswlib

A LlamaIndex vector store using Hnswlib, a header-only C++ HNSW implementation with python bindings.

## Usage

Pre-requisite:

```
pip install llama-index-vector-stores-hnswlib
pip install hnswlib
```

A minimal example:

```python
import hnswlib
from llama_index.vector_stores_hnswlib import HnswlibVectorStore

space = "ip"  # distance function
dim = 768  # embedding dimension
hnswlib_index = hnswlib.Index(space, dim)
hnswlib_index.init_index(max_elements=10)

hnsw_vector_store = HnswlibVectorStore(hnswlib_index=index)
```

## More examples and references

A detailed usage guede can be found [in this demo notebook](https://docs.llamaindex.ai/en/stable/examples/vector_stores/HnswlibIndexDemo.html) in the LlamaIndex docs.

Hnswlib documentation and implementation can be found [here](https://github.com/nmslib/hnswlib).
