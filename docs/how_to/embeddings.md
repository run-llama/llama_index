# Embedding support

GPT Index provides support for embeddings in the following format:
- Adding embeddings to Document objects
- Using a Vector Store as an underlying index (e.g. `GPTSimpleVectorIndex`, `GPTFaissIndex`)
- Querying our list and tree indices with embeddings.

## Adding embeddings to Document objects

You can pass in user-specified embeddings when constructing an index. This gives you control
in specifying embeddings per Document instead of having us determine embeddings for your text (see below).

Simply specify the `embedding` field when creating a Document:

![](/_static/embeddings/doc_example.jpeg)

## Using a Vector Store as an Underlying Index

<!-- Please see the corresponding section in our [Vector Stores](/how_to/vector_stores.md#loading-data-from-vector-stores-using-data-connector) -->
Please see the corresponding section in our [Vector Stores](/how_to/vector_stores.md)
guide for more details.

## Using an Embedding Query Mode in List/Tree Index

GPT Index provides embedding support to our tree and list indices. In addition to each node storing text, each node can optionally store an embedding.
During query-time, we can use embeddings to do max-similarity retrieval of nodes before calling the LLM to synthesize an answer. 
Since similarity lookup using embeddings (e.g. using cosine similarity) does not require a LLM call, embeddings serve as a cheaper lookup mechanism instead
of using LLMs to traverse nodes.

#### How are Embeddings Generated?

Since we offer embedding support during *query-time* for our list and tree indices, 
embeddings are lazily generated and then cached (if `mode="embedding"` is specified during `index.query(...)`), and not during index construction.
This design choice prevents the need to generate embeddings for all text chunks during index construction.

NOTE: Our [vector-store based indices](/how_to/vector_stores.md) generate embeddings during index construction.

#### Embedding Lookups
For the list index (`GPTListIndex`):
- We iterate through every node in the list, and identify the top k nodes through embedding similarity. We use these nodes to synthesize an answer.
- See the [List Query API](/reference/indices/list_query.rst) for more details.
- NOTE: the embedding-mode usage of the list index is roughly equivalent with the usage of our `GPTSimpleVectorIndex`; the main
    difference is when embeddings are generated (during query-time for the list index vs. index construction for the simple vector index).

For the tree index (`GPTTreeIndex`):
- We start with the root nodes, and traverse down the tree by picking the child node through embedding similarity.
- See the [Tree Query API](/reference/indices/tree_query.rst) for more details.

**Example Notebook**

An example notebook is given [here](https://github.com/jerryjliu/gpt_index/blob/main/examples/test_wiki/TestNYC_Embeddings.ipynb).



(custom-embeddings)=
## Custom Embeddings

GPT Index allows you to define custom embedding modules. By default, we use `text-embedding-ada-002` from OpenAI. 

You can also choose to plug in embeddings from
Langchain's [embeddings](https://langchain.readthedocs.io/en/latest/reference/modules/embeddings.html) module.
We introduce a wrapper class, 
[`LangchainEmbedding`](/reference/embeddings.rst), for integration into GPT Index.

An example snippet is shown below (to use Hugging Face embeddings) on the GPTListIndex:

```python
from gpt_index import GPTListIndex, SimpleDirectoryReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from gpt_index import LangchainEmbedding

# load in HF embedding model from langchain
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

# load index
new_index = GPTListIndex.load_from_disk('index_list_emb.json')

# query with embed_model specified
response = new_index.query(
    "<query_text>", 
    mode="embedding", 
    verbose=True, 
    embed_model=embed_model
)
print(response)
```

Another example snippet is shown for GPTSimpleVectorIndex.

```python
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from gpt_index import LangchainEmbedding

# load in HF embedding model from langchain
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

# load index
new_index = GPTSimpleVectorIndex.load_from_disk(
    'index_simple_vector.json', 
    embed_model=embed_model
)

# query will use the same embed_model
response = new_index.query(
    "<query_text>", 
    mode="default", 
    verbose=True, 
)
print(response)
```
