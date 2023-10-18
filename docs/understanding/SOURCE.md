# Basic Usage Pattern

The general usage pattern of LlamaIndex is as follows:

1. Load in documents (either manually, or through a data loader)
2. Parse the Documents into Nodes
3. Construct Index (from Nodes or Documents)
4. [Optional, Advanced] Building indices on top of other indices
5. Query the index
6. Parsing the response




### Reusing Nodes across Index Structures

If you have multiple Node objects defined, and wish to share these Node
objects across multiple index structures, you can do that.
Simply instantiate a StorageContext object,
add the Node objects to the underlying DocumentStore,
and pass the StorageContext around.

```python
from llama_index import StorageContext

storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

index1 = VectorStoreIndex(nodes, storage_context=storage_context)
index2 = SummaryIndex(nodes, storage_context=storage_context)
```

**NOTE**: If the `storage_context` argument isn't specified, then it is implicitly
created for each index during index construction. You can access the docstore
associated with a given index through `index.storage_context`.

### Inserting Documents or Nodes

You can also take advantage of the `insert` capability of indices to insert Document objects
one at a time instead of during index construction.

```python
from llama_index import VectorStoreIndex

index = VectorStoreIndex([])
for doc in documents:
    index.insert(doc)
```

If you want to insert nodes on directly you can use `insert_nodes` function
instead.

```python
from llama_index import VectorStoreIndex

# nodes: Sequence[Node]
index = VectorStoreIndex([])
index.insert_nodes(nodes)
```

See the [Document Management How-To](/core_modules/data_modules/index/document_management.md) for more details on managing documents and an example notebook.




### Customizing Prompts

Depending on the index used, we used default prompt templates for constructing the index (and also insertion/querying).
See [Custom Prompts How-To](/core_modules/model_modules/prompts.md) for more details on how to customize your prompt.

### Cost Analysis

Creating an index, inserting to an index, and querying an index may use tokens. We can track
token usage through the outputs of these operations. When running operations,
the token usage will be printed.

You can also fetch the token usage through `TokenCountingCallback` handler.
See [Cost Analysis How-To](/core_modules/supporting_modules/cost_analysis/usage_pattern.md) for more details.

## 5. Query the index.

After building the index, you can now query it with a `QueryEngine`. Note that a "query" is simply an input to an LLM -
this means that you can use the index for question-answering, but you can also do more than that!

### High-level API

To start, you can query an index with the default `QueryEngine` (i.e., using default configs), as follows:

```python
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)

response = query_engine.query("Write an email to the user given their background information.")
print(response)
```

### Low-level API

We also support a low-level composition API that gives you more granular control over the query logic.
Below we highlight a few of the possible customizations.

```python
from llama_index import (
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor

# build index
index = VectorStoreIndex.from_documents(documents)

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=2,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.7)
    ]

)

# query
response = query_engine.query("What did the author do growing up?")
print(response)
```

You may also add your own retrieval, response synthesis, and overall query logic, by implementing the corresponding interfaces.

For a full list of implemented components and the supported configurations, please see the detailed [reference docs](/api_reference/query.rst).

In the following, we discuss some commonly used configurations in detail.

### Configuring retriever

An index can have a variety of index-specific retrieval modes.
For instance, a summary index supports the default `SummaryIndexRetriever` that retrieves all nodes, and
`SummaryIndexEmbeddingRetriever` that retrieves the top-k nodes by embedding similarity.

For convenience, you can also use the following shorthand:

```python
    # SummaryIndexRetriever
    retriever = index.as_retriever(retriever_mode='default')
    # SummaryIndexEmbeddingRetriever
    retriever = index.as_retriever(retriever_mode='embedding')
```

After choosing your desired retriever, you can construct your query engine:

```python
query_engine = RetrieverQueryEngine(retriever)
response = query_engine.query("What did the author do growing up?")
```

The full list of retrievers for each index (and their shorthand) is documented in the [Query Reference](/api_reference/query.rst).

(setting-response-mode)=

### Configuring response synthesis

After a retriever fetches relevant nodes, a `BaseSynthesizer` synthesizes the final response by combining the information.

You can configure it via

```python
query_engine = RetrieverQueryEngine.from_args(retriever, response_mode=<response_mode>)
```

Right now, we support the following options:

- `default`: "create and refine" an answer by sequentially going through each retrieved `Node`;
  This makes a separate LLM call per Node. Good for more detailed answers.
- `compact`: "compact" the prompt during each LLM call by stuffing as
  many `Node` text chunks that can fit within the maximum prompt size. If there are
  too many chunks to stuff in one prompt, "create and refine" an answer by going through
  multiple prompts.
- `tree_summarize`: Given a set of `Node` objects and the query, recursively construct a tree
  and return the root node as the response. Good for summarization purposes.
- `no_text`: Only runs the retriever to fetch the nodes that would have been sent to the LLM,
  without actually sending them. Then can be inspected by checking `response.source_nodes`.
  The response object is covered in more detail in Section 5.
- `accumulate`: Given a set of `Node` objects and the query, apply the query to each `Node` text
  chunk while accumulating the responses into an array. Returns a concatenated string of all
  responses. Good for when you need to run the same query separately against each text
  chunk.

```python
index = SummaryIndex.from_documents(documents)
retriever = index.as_retriever()

# default
query_engine = RetrieverQueryEngine.from_args(retriever, response_mode='default')
response = query_engine.query("What did the author do growing up?")

# compact
query_engine = RetrieverQueryEngine.from_args(retriever, response_mode='compact')
response = query_engine.query("What did the author do growing up?")

# tree summarize
query_engine = RetrieverQueryEngine.from_args(retriever, response_mode='tree_summarize')
response = query_engine.query("What did the author do growing up?")

# no text
query_engine = RetrieverQueryEngine.from_args(retriever, response_mode='no_text')
response = query_engine.query("What did the author do growing up?")
```

### Configuring node postprocessors (i.e. filtering and augmentation)

We also support advanced `Node` filtering and augmentation that can further improve the relevancy of the retrieved `Node` objects.
This can help reduce the time/number of LLM calls/cost or improve response quality.

For example:

- `KeywordNodePostprocessor`: filters nodes by `required_keywords` and `exclude_keywords`.
- `SimilarityPostprocessor`: filters nodes by setting a threshold on the similarity score (thus only supported by embedding-based retrievers)
- `PrevNextNodePostprocessor`: augments retrieved `Node` objects with additional relevant context based on `Node` relationships.

The full list of node postprocessors is documented in the [Node Postprocessor Reference](/api_reference/node_postprocessor.rst).

To configure the desired node postprocessors:

```python
node_postprocessors = [
    KeywordNodePostprocessor(
        required_keywords=["Combinator"],
        exclude_keywords=["Italy"]
    )
]
query_engine = RetrieverQueryEngine.from_args(
    retriever, node_postprocessors=node_postprocessors
)
response = query_engine.query("What did the author do growing up?")
```

## 6. Parsing the response

The object returned is a [`Response` object](/api_reference/response.rst).
The object contains both the response text as well as the "sources" of the response:

```python
response = query_engine.query("<query_str>")

# get response
# response.response
str(response)

# get sources
response.source_nodes
# formatted sources
response.get_formatted_sources()
```

An example is shown below.
![](/_static/response/response_1.jpeg)
