# Using Managed Indices

LlamaIndex offers multiple integration points with Managed Indices. A managed index is a special type of index that is not managed locally as part of LlamaIndex but instead is managed via an API, such as Vectara.

(managed-index)=

## Using a Managed Index

Similar to any other index within LlamaIndex (tree, keyword table, list), any `ManagedIndex` can be constructed with a collection
of documents. Once constructed, the index can be used for querying.

If the Index has been previously populated with documents - it can also be used directly for querying.

`VectaraIndex` is currently the only supported managed index, although we expect more to be available soon.
Below we show how to use it.

**Vectara Index Construction/Querying**

Use the [Vectara Console](https://console.vectara.com/login) to create a corpus, and add an API key for access. 
Then put the customer id, corpus id, and API key in your environment as shown below.

Then construct the Vectara Index and query it as follows:

```python
from llama_index import ManagedIndex, SimpleDirectoryReade
from llama_index.managed import VectaraIndex

# Load documents and build index
vectara_customer_id = os.environ.get("VECTARA_CUSTOMER_ID")
vectara_corpus_id = os.environ.get("VECTARA_CORPUS_ID")
vectara_api_key = os.environ.get("VECTARA_API_KEY")
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = VectaraIndex.from_documents(documents, vectara_customer_id=vectara_customer_id, vectara_corpus_id=vectara_corpus_id, vectara_api_key=vectara_api_key)

# Query index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

# Query with metadata
filters = MetadataFilters(filters=[ExactMatchFilter(key="theme", value="Mafia")])
retriever = index.as_retriever(filters=filters)
result = retriever.retrieve("What is inception about?")
```


```{toctree}
---
caption: Examples
maxdepth: 1
---
../../examples/vector_stores/VectaraDemo.ipynb
```
