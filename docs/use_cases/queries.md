# Queries over your Data

At a high-level, LlamaIndex gives you the ability to query your data for any downstream LLM use case,
whether it's question-answering, summarization, or a component in a chatbot.

This section describes the different ways you can query your data with LlamaIndex, roughly in order
of simplest (top-k semantic search), to more advanced capabilities.

### Semantic Search 

The most basic example usage of LlamaIndex is through semantic search. We provide
a simple in-memory vector store for you to get started, but you can also choose
to use any one of our [vector store integrations](/how_to/integrations/vector_stores.md):

```python
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader('data').load_data()
index = GPTVectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)

```

**Tutorials**
- [Quickstart](/getting_started/starter_example.md)

**Guides**
- [Example](../examples/vector_stores/SimpleIndexDemo.ipynb) ([Notebook](https://github.com/jerryjliu/llama_index/tree/main/docs/examples/vector_stores/SimpleIndexDemo.ipynb))


### Summarization

A summarization query requires the LLM to iterate through many if not most documents in order to synthesize an answer.
For instance, a summarization query could look like one of the following: 
- "What is a summary of this collection of text?"
- "Give me a summary of person X's experience with the company."

In general, a list index would be suited for this use case. A list index by default goes through all the data.

Empirically, setting `response_mode="tree_summarize"` also leads to better summarization results.

```python
index = GPTListIndex.from_documents(documents)

query_engine = index.as_query_engine(
    response_mode="tree_summarize"
)
response = query_engine.query("<summarization_query>")
```

### Queries over Structured Data

LlamaIndex supports queries over structured data, whether that's a Pandas DataFrame or a SQL Database.

Here are some relevant resources:

**Tutorials**

- [Guide on Text-to-SQL](/guides/tutorials/sql_guide.md)

**Guides**
- [SQL Guide (Core)](../examples/index_structs/struct_indices/SQLIndexDemo.ipynb) ([Notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/index_structs/struct_indices/SQLIndexDemo.ipynb))
- [SQL Guide (Setting Context)](../examples/index_structs/struct_indices/SQLIndexDemo-Context.ipynb) ([Notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/index_structs/struct_indices/SQLIndexDemo-Context.ipynb))
- [SQL Guide (Many Tables)](../examples/index_structs/struct_indices//SQLIndexDemo-ManyTables.ipynb) ([Notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/index_structs/struct_indices/SQLIndexDemo-ManyTables.ipynb))
- [Pandas Demo](../examples/index_structs/struct_indices/PandasIndexDemo.ipynb) ([Notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/index_structs/struct_indices/PandasIndexDemo.ipynb))


### Synthesis over Heterogeneous Data

LlamaIndex supports synthesizing across heterogeneous data sources. This can be done by composing a graph over your existing data.
Specifically, compose a list index over your subindices. A list index inherently combines information for each node; therefore
it can synthesize information across your heterogeneous data sources.

```python
from llama_index import GPTVectorStoreIndex, GPTListIndex
from llama_index.indices.composability import ComposableGraph

index1 = GPTVectorStoreIndex.from_documents(notion_docs)
index2 = GPTVectorStoreIndex.from_documents(slack_docs)

graph = ComposableGraph.from_indices(GPTListIndex, [index1, index2], index_summaries=["summary1", "summary2"])
query_engine = graph.as_query_engine()
response = query_engine.query("<query_str>")

```

**Guides**
- [Composability](/how_to/index_structs/composability.md)
- [City Analysis](../examples/composable_indices/city_analysis/PineconeDemo-CityAnalysis.ipynb) ([Notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/composable_indices/city_analysis/PineconeDemo-CityAnalysis.ipynb))



### Routing over Heterogeneous Data

LlamaIndex also supports routing over heterogeneous data sources with `RouterQueryEngine` - for instance, if you want to "route" a query to an 
underlying Document or a sub-index.


To do this, first build the sub-indices over different data sources.
Then construct the corresponding query engines, and give each query engine a description to obtain a `QueryEngineTool`.

```python
from llama_index import GPTTreeIndex, GPTVectorStoreIndex
from llama_index.tools import QueryEngineTool

...

# define sub-indices
index1 = GPTVectorStoreIndex.from_documents(notion_docs)
index2 = GPTVectorStoreIndex.from_documents(slack_docs)

# define query engines and tools
tool1 = QueryEngineTool.from_defaults(
    query_engine=index1.as_query_engine(), 
    description="Use this query engine to do...",
)
tool2 = QueryEngineTool.from_defaults(
    query_engine=index2.as_query_engine(), 
    description="Use this query engine for something else...",
)
```

Then, we define a `RouterQueryEngine` over them.
By default, this uses a `LLMSingleSelector` as the router, which uses the LLM to choose the best sub-index to router the query to, given the descriptions.

```python
from llama_index.query_engine import RouterQueryEngine

query_engine = RouterQueryEngine.from_defaults(
    query_engine_tools=[tool1, tool2]
)

response = query_engine.query(
    "In Notion, give me a summary of the product roadmap."
)

```

**Guides**
- [Router Query Engine Guide](../examples/query_engine/RouterQueryEngine.ipynb) ([Notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/query_engine/RouterQueryEngine.ipynb))
- [City Analysis Unified Query Interface](../examples/composable_indices/city_analysis/City_Analysis-Unified-Query.ipynb) ([Notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/composable_indices/city_analysis/PineconeDemo-CityAnalysis.ipynb))

### Compare/Contrast Queries
You can explicitly perform compare/contrast queries with a **query transformation** module within a ComposableGraph.

```python
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
decompose_transform = DecomposeQueryTransform(
    llm_predictor_chatgpt, verbose=True
)
```

This module will help break down a complex query into a simpler one over your existing index structure.

**Guides**
- [Query Transformations](/how_to/query/query_transformations.md)
- [City Analysis Compare/Contrast Example](../examples//composable_indices/city_analysis/City_Analysis-Decompose.ipynb) ([Notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/composable_indices/city_analysis/City_Analysis-Decompose.ipynb))

You can also rely on the LLM to *infer* whether to perform compare/contrast queries (see Multi-Document Queries below).

### Multi-Document Queries

Besides the explicit synthesis/routing flows described above, LlamaIndex can support more general multi-document queries as well. 
It can do this through our `SubQuestionQueryEngine` class. Given a query, this query engine will generate a "query plan" containing
sub-queries against sub-documents before synthesizing the final answer.

To do this, first define an index for each document/data source, and wrap it with a `QueryEngineTool` (similar to above):

```python
from llama_index.tools import QueryEngineTool, ToolMetadata

query_engine_tools = [
    QueryEngineTool(
        query_engine=sept_engine, 
        metadata=ToolMetadata(name='sept_22', description='Provides information about Uber quarterly financials ending September 2022')
    ),
    QueryEngineTool(
        query_engine=june_engine, 
        metadata=ToolMetadata(name='june_22', description='Provides information about Uber quarterly financials ending June 2022')
    ),
    QueryEngineTool(
        query_engine=march_engine, 
        metadata=ToolMetadata(name='march_22', description='Provides information about Uber quarterly financials ending March 2022')
    ),
]
```

Then, we define a `SubQuestionQueryEngine` over these tools:

```python
from llama_index.query_engine import SubQuestionQueryEngine

query_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)

```

This query engine can execute any number of sub-queries against any subset of query engine tools before synthesizing the final answer.
This makes it especially well-suited for compare/contrast queries across documents as well as queries pertaining to a specific document.

**Guides**
- [Sub Question Query Engine (Intro)](../examples/query_engine/sub_question_query_engine.ipynb)
- [10Q Analysis (Uber)](../examples/usecases/10q_sub_question.ipynb)
- [10K Analysis (Uber and Lyft)](../examples/usecases/10k_sub_question.ipynb)


### Multi-Step Queries

LlamaIndex can also support iterative multi-step queries. Given a complex query, break it down into an initial subquestions,
and sequentially generate subquestions based on returned answers until the final answer is returned.

For instance, given a question "Who was in the first batch of the accelerator program the author started?",
the module will first decompose the query into a simpler initial question "What was the accelerator program the author started?",
query the index, and then ask followup questions.

**Guides**
- [Query Transformations](/how_to/query/query_transformations.md)
- [Multi-Step Query Decomposition](../examples/query_transformations/HyDEQueryTransformDemo.ipynb) ([Notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/query_transformations/HyDEQueryTransformDemo.ipynb))


### Temporal Queries

LlamaIndex can support queries that require an understanding of time. It can do this in two ways:
- Decide whether the query requires utilizing temporal relationships between nodes (prev/next relationships) in order to retrieve additional context to answer the question.
- Sort by recency and filter outdated context.

**Guides**
- [Second-Stage Postprocessing Guide](/how_to/query/second_stage.md)
- [Prev/Next Postprocessing](../examples/node_postprocessor/PrevNextPostprocessorDemo.ipynb)
- [Recency Postprocessing](../examples/node_postprocessor/RecencyPostprocessorDemo.ipynb)

