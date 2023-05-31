# Query Transformations


LlamaIndex allows you to perform *query transformations* over your index structures.
Query transformations are modules that will convert a query into another query. They can be **single-step**, as in the transformation is run once before the query is executed against an index. 

They can also be **multi-step**, as in: 
1. The query is transformed, executed against an index, 
2. The response is retrieved.
3. Subsequent queries are transformed/executed in a sequential fashion.

We list some of our query transformations in more detail below.

#### Use Cases
Query transformations have multiple use cases:
- Transforming an initial query into a form that can be more easily embedded (e.g. HyDE)
- Transforming an initial query into a subquestion that can be more easily answered from the data (single-step query decomposition)
- Breaking an initial query into multiple subquestions that can be more easily answered on their own. (multi-step query decomposition)


### HyDE (Hypothetical Document Embeddings)

[HyDE](http://boston.lti.cs.cmu.edu/luyug/HyDE/HyDE.pdf) is a technique where given a natural language query, a hypothetical document/answer is generated first. This hypothetical document is then used for embedding lookup rather than the raw query.

To use HyDE, an example code snippet is shown below.

```python
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.indices.query.query_transform.base import HyDEQueryTransform
from llama_index.indices.query import TransformQueryEngine

# load documents, build index
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = GPTVectorStoreIndex(documents)

# run query with HyDE query transform
query_str = "what did paul graham do after going to RISD"
hyde = HyDEQueryTransform(include_original=True)
query_engine = index.as_query_engine()
query_engine = TransformQueryEngine(query_engine, query_transform=hyde)
response = query_engine.query(query_str)
print(response)

```

Check out our [example notebook](../../examples/query_transformations/HyDEQueryTransformDemo.ipynb) for a full walkthrough.


### Single-Step Query Decomposition

Some recent approaches (e.g. [self-ask](https://ofir.io/self-ask.pdf), [ReAct](https://arxiv.org/abs/2210.03629)) have suggested that LLM's 
perform better at answering complex questions when they break the question into smaller steps. We have found that this is true for queries that require knowledge augmentation as well.

If your query is complex, different parts of your knowledge base may answer different "subqueries" around the overall query.

Our single-step query decomposition feature transforms a **complicated** question into a simpler one over the data collection to help provide a sub-answer to the original question.

This is especially helpful over a [composed graph](/how_to/index_structs/composability.md). Within a composed graph, a query can be routed to multiple subindexes, each representing a subset of the overall knowledge corpus. Query decomposition allows us to transform the query into a more suitable question over any given index.

An example image is shown below.

![](/_static/query_transformations/single_step_diagram.png)


Here's a corresponding example code snippet over a composed graph.

```python

# Setting: a list index composed over multiple vector indices
# llm_predictor_chatgpt corresponds to the ChatGPT LLM interface
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
decompose_transform = DecomposeQueryTransform(
    llm_predictor_chatgpt, verbose=True
)

# initialize indexes and graph
...


# configure retrievers
vector_query_engine = vector_index.as_query_engine()
vector_query_engine = TransformQueryEngine(
    vector_query_engine, 
    query_transform=decompose_transform
    transform_extra_info={'index_summary': vector_index.index_struct.summary}
)
custom_query_engines = {
    vector_index.index_id: vector_query_engine
} 

# query
query_str = (
    "Compare and contrast the airports in Seattle, Houston, and Toronto. "
)
query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)
response = query_engine.query(query_str)
```

Check out our [example notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/composable_indices/city_analysis/City_Analysis-Decompose.ipynb) for a full walkthrough.



### Multi-Step Query Transformations

Multi-step query transformations are a generalization on top of existing single-step query transformation approaches.

Given an initial, complex query, the query is transformed and executed against an index. The response is retrieved from the query. 
Given the response (along with prior responses) and the query, followup questions may be asked against the index as well. This technique allows a query to be run against a single knowledge source until that query has satisfied all questions.

An example image is shown below.

![](/_static/query_transformations/multi_step_diagram.png)


Here's a corresponding example code snippet.

```python
from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform
# gpt-4
step_decompose_transform = StepDecomposeQueryTransform(
    llm_predictor, verbose=True
)

query_engine = index.as_query_engine()
query_engine = MultiStepQueryEngine(query_engine, query_transform=step_decompose_transform)

response = query_engine.query(
    "Who was in the first batch of the accelerator program the author started?",
)
print(str(response))

```

Check out our [example notebook](https://github.com/jerryjliu/llama_index/blob/main/examples/vector_indices/SimpleIndexDemo-multistep.ipynb) for a full walkthrough.


```{toctree}
---
caption: Examples
maxdepth: 1
---
../../examples/query_transformations/HyDEQueryTransformDemo.ipynb
../../examples/query_transformations/SimpleIndexDemo-multistep.ipynb
```