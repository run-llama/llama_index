# ObjectBox VectorStore For LlamaIndex

### About

This package contains the [ObjectBox](https://objectbox.io/) integrations for [LlamaIndex](https://www.llamaindex.ai/)

### Getting Started

Install the `llama-index-vector-stores-objectbox` package from PyPI via pip.

```commandline
pip install llama-index-vector-stores-objectbox
```

You can import the ObjectBox vector-store with `from llama_index.vector_stores.objectbox import ObjectBoxVectorStore` and start using it,

```python
from llama_index.vector_stores.objectbox import ObjectBoxVectorStore
from objectbox import VectorDistanceType

embedding_dim = 384  # size of the embeddings to be stored

vector_store = ObjectBoxVectorStore(
    embedding_dim,
    distance_type=VectorDistanceType.COSINE,
    db_directory="obx_data",
    clear_db=False,
    do_log=True,
)
```

- `embedding_dim` (required): The dimensions of the embeddings that the vector DB will hold
- `distance_type`: Choose from `COSINE`, `DOT_PRODUCT`, `DOT_PRODUCT_NON_NORMALIZED` and `EUCLIDEAN`
- `db_directory`: The path of the directory where the `.mdb` ObjectBox database file should be created
- `clear_db`: Deletes the existing database file if it exists on `db_directory`
- `do_log`: Enables logging from the ObjectBox integration

### A complete RAG example

Along the `llama-index-vector-stores-objectbox`, install the following packages,

```commandline
pip install llama-index --quiet
pip install llama-index-embeddings-huggingface --quiet
pip install llama-index-llms-gemini --quiet
```

Download a sample text file,

```commandline
mkdir -p 'data/paul_graham/'
wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'
```

This example will require a Gemini API key. You can get an API-key from the [Gemini developer console](https://aistudio.google.com/app/apikey). Execute the following Python script to generate an answer for `Who is Paul Graham?` from the text file,

```python
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.objectbox import ObjectBoxVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from objectbox import VectorDistanceType
import getpass

gemini_key_api = getpass.getpass("Gemini API Key: ")
gemini_llm = Gemini(api_key=gemini_key_api)

# Configure embedding model from HuggingFace
hf_embedding = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
embedding_dim = 384

# Setup file reader and text splitter
reader = SimpleDirectoryReader("./data/paul_graham")
documents = reader.load_data()

node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents)

# Configure ObjectBox as a vector-store
vector_store = ObjectBoxVectorStore(
    embedding_dim,
    distance_type=VectorDistanceType.COSINE,
    db_directory="obx_data",
    clear_db=False,
    do_log=True,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

Settings.llm = gemini_llm
Settings.embed_model = hf_embedding

index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

query_engine = index.as_query_engine()
response = query_engine.query("Who is Paul Graham?")
print(response)
```

### License

```text
MIT License

Copyright (c) 2024 ObjectBox, Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
