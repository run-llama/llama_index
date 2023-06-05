# Embedding support

LlamaIndex provides support for embeddings in the following format:
- Adding embeddings to Document objects
- Using a Vector Store as an underlying index (e.g. `GPTVectorStoreIndex`)
- Querying our list and tree indices with embeddings.

## Adding embeddings to Document objects

You can pass in user-specified embeddings when constructing an index. This gives you control
in specifying embeddings per Document instead of having us determine embeddings for your text (see below).

Simply specify the `embedding` field when creating a Document:

![](/_static/embeddings/doc_example.jpeg)

## Using a Vector Store as an Underlying Index

<!-- Please see the corresponding section in our [Vector Stores](/how_to/vector_stores.md#loading-data-from-vector-stores-using-data-connector) -->
Please see the corresponding section in our [Vector Stores](/how_to/integrations/vector_stores.md)
guide for more details.

## Using an Embedding Query Mode in List/Tree Index

LlamaIndex provides embedding support to our tree and list indices. In addition to each node storing text, each node can optionally store an embedding.
During query-time, we can use embeddings to do max-similarity retrieval of nodes before calling the LLM to synthesize an answer. 
Since similarity lookup using embeddings (e.g. using cosine similarity) does not require a LLM call, embeddings serve as a cheaper lookup mechanism instead
of using LLMs to traverse nodes.

#### How are Embeddings Generated?

Since we offer embedding support during *query-time* for our list and tree indices, 
embeddings are lazily generated and then cached (if `retriever_mode="embedding"` is specified during `query(...)`), and not during index construction.
This design choice prevents the need to generate embeddings for all text chunks during index construction.

NOTE: Our [vector-store based indices](/how_to/integrations/vector_stores.md) generate embeddings during index construction.

#### Embedding Lookups
For the list index (`ListIndex`):
- We iterate through every node in the list, and identify the top k nodes through embedding similarity. We use these nodes to synthesize an answer.
- See the [List Retriever API](/reference/query/retrievers/list.rst) for more details.
- NOTE: the embedding-mode usage of the list index is roughly equivalent with the usage of our `GPTVectorStoreIndex`; the main
    difference is when embeddings are generated (during query-time for the list index vs. index construction for the simple vector index).

For the tree index (`GPTTreeIndex`):
- We start with the root nodes, and traverse down the tree by picking the child node through embedding similarity.
- See the [Tree Query API](/reference/query/retrievers/tree.rst) for more details.

**Example Notebook**

An example notebook is given [here](https://github.com/jerryjliu/llama_index/blob/main/examples/test_wiki/TestNYC_Embeddings.ipynb).



(custom-embeddings)=
## Custom Embeddings

LlamaIndex allows you to define custom embedding modules. By default, we use `text-embedding-ada-002` from OpenAI. 

You can also choose to plug in embeddings from
Langchain's [embeddings](https://langchain.readthedocs.io/en/latest/reference/modules/embeddings.html) module.
We introduce a wrapper class, 
[`LangchainEmbedding`](/reference/service_context/embeddings.rst), for integration into LlamaIndex.

An example snippet is shown below (to use Hugging Face embeddings) on the ListIndex:

```python
from llama_index import ListIndex, SimpleDirectoryReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext

# load in HF embedding model from langchain
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# build index
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
new_index = ListIndex.from_documents(documents)

# query with embed_model specified
query_engine = new_index.as_query_engine(
    retriever_mode="embedding", 
    verbose=True, 
    service_context=service_context
)
response = query_engine.query("<query_text>")
print(response)
```

Another example snippet is shown for GPTVectorStoreIndex.

```python
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext

# load in HF embedding model from langchain
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# load index
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
new_index = GPTVectorStoreIndex.from_documents(
    documents, 
    service_context=service_context,
)

# query will use the same embed_model
query_engine = new_index.as_query_engine(
    verbose=True, 
)
response = query_engine.query("<query_text>")
print(response)
```
