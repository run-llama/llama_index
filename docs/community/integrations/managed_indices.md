# Using Managed Indices

LlamaIndex offers multiple integration points with Managed Indices. A managed index is a special type of index that is not managed locally as part of LlamaIndex but instead is managed via an API, such as [Vectara](https://vectara.com).

## Using a Managed Index

Similar to any other index within LlamaIndex (tree, keyword table, list), any `ManagedIndex` can be constructed with a collection
of documents. Once constructed, the index can be used for querying.

If the Index has been previously populated with documents - it can also be used directly for querying.

## Google Generative Language Semantic Retriever.

Google's Semantic Retrieve provides both querying and retrieval capabilities. Create a managed index, insert documents, and use a query engine or retriever anywhere in LlamaIndex!

```python
from llama_index import SimpleDirectoryReader
from llama_index.indices.managed.google.generativeai import GoogleIndex

# Create a corpus
index = GoogleIndex.create_corpus(display_name="My first corpus!")
print(f"Newly created corpus ID is {index.corpus_id}.")

# Ingestion
documents = SimpleDirectoryReader("data").load_data()
index.insert_documents(documents)

# Querying
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

# Retrieving
retriever = index.as_retriever()
source_nodes = retriever.retrieve("What did the author do growing up?")
```

See the notebook guide for full details.

```{toctree}
---
caption: Examples
maxdepth: 1
---
/examples/managed/GoogleDemo.ipynb
```

## Vectara

First, [sign up](https://vectara.com/integrations/llama_index) and use the Vectara Console to create a corpus (aka Index), and add an API key for access.
Then put the customer id, corpus id, and API key in your environment.

Then construct the Vectara Index and query it as follows:

```python
from llama_index import ManagedIndex, SimpleDirectoryReade
from llama_index.indices import VectaraIndex

# Load documents and build index
vectara_customer_id = os.environ.get("VECTARA_CUSTOMER_ID")
vectara_corpus_id = os.environ.get("VECTARA_CORPUS_ID")
vectara_api_key = os.environ.get("VECTARA_API_KEY")
documents = SimpleDirectoryReader("../paul_graham_essay/data").load_data()
index = VectaraIndex.from_documents(
    documents,
    vectara_customer_id=vectara_customer_id,
    vectara_corpus_id=vectara_corpus_id,
    vectara_api_key=vectara_api_key,
)

# Query index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
```

Note that if the environment variables `VECTARA_CUSTOMER_ID`, `VECTARA_CORPUS_ID` and `VECTARA_API_KEY` are in the environment already, you do not have to explicitly specifying them in your call and the VectaraIndex class will read them from the environment. For example this should be equivalent to the above, if these variables are in the environment already:

```python
from llama_index import ManagedIndex, SimpleDirectoryReade
from llama_index.indices import VectaraIndex

# Load documents and build index
documents = SimpleDirectoryReader("../paul_graham_essay/data").load_data()
index = VectaraIndex.from_documents(documents)

# Query index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
```

```{toctree}
---
caption: Examples
maxdepth: 1
---
/examples/managed/vectaraDemo.ipynb
```
