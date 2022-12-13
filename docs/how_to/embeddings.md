# Embedding support

GPT Index provides embedding support to our tree and list indices. In addition to each node storing text, each node can optionally store an embedding.
During query-time, we can use embeddings to do max-similarity retrieval of nodes before calling the LLM to synthesize an answer. 
Since similarity lookup using embeddings (e.g. using cosine similarity) does not require a LLM call, embeddings serve as a cheaper lookup mechanism instead
of using LLMs to traverse nodes.

NOTE: we currently support OpenAI embeddings. External embeddings are coming soon!

**How are Embeddings Generated?**

Embeddings are lazily generated and then cached at query time (if mode="embedding" is specified during `index.query`), and not during index construction.
This design choice prevents the need to generate embeddings for all text chunks during index construction.

**Embedding Lookups**
For the list index:
- We iterate through every node in the list, and identify the top k nodes through embedding similarity. We use these nodes to synthesize an answer.
- See the [List Query API](/reference/indices/list_query.rst) for more details.

For the tree index:
- We start with the root nodes, and traverse down the tree by picking the child node through embedding similarity.
- See the [Tree Query API](/reference/indices/tree_query.rst) for more details.

**Example Notebook**

An example notebook is given [here](https://github.com/jerryjliu/gpt_index/blob/main/examples/test_wiki/TestNYC_Embeddings.ipynb).

