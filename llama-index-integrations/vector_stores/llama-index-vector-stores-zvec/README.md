# LlamaIndex Vector_Stores Integration: Zvec

[Zvec](https://vespa.ai/) is an open-source, fast, lightweight, and feature-rich vector database that runs entirely in-process — no server, daemon, or external infrastructure required.

## 💫 Features

- **Blazing Fast**: Searches billions of vectors in milliseconds.
- **Simple, Just Works**: Install with `pip install zvec` and start searching in seconds. No servers, no config, no fuss.
- **Dense + Sparse Vectors**: Work with both dense and sparse embeddings, with native support for multi-vector queries in a single call.
- **Hybrid Search**: Combine semantic similarity with structured filters for precise results.
- **Runs Anywhere**: As an in-process library, Zvec runs wherever your code runs — notebooks, servers, CLI tools, or even edge devices.

## 🚀 Installation

```bash
pip install llama-index-core zvec
```

## Usage

### download data

```Bash
mkdir -p 'data/paul_graham/'
wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'
```

### example

```Python
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.zvec import ZvecVectorStore

vector_store = ZvecVectorStore(
    path="quickstart.zvec",
    collection_name="quickstart",
    embed_dim=1536
)

from llama_index.core import SimpleDirectoryReader
from IPython.display import Markdown, display

# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

# build index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# query
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
```
