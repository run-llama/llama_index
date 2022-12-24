# Embedding support

GPT Index provides support for embeddings in the following format:
- Using a Vector Store as an underlying index (e.g. `GPTSimpleVectorIndex`, `GPTFaissIndex`)
- Querying our list and tree indices with embeddings.

NOTE: we currently support OpenAI embeddings. External embeddings are coming soon!

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

