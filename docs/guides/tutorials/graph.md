# A Guide to Creating a Unified Query Framework over your Indexes

LlamaIndex offers a variety of different [query use cases](/docs/use_cases/queries.md). 

For simple queries, we may want to use a single index data structure, such as a `GPTSimpleVectorIndex` for semantic search, or `GPTListIndex` for summarization.

For more complex queries, we may want to use a composable graph. 

But how do we integrate indexes and graphs into our LLM application? Different indexes and graphs may be better suited for different types of queries that you may want to run. 

In this guide, we show how you can unify the diverse use cases of different index/graph structures under a **single** query framework.

### Setup

In this example, we will analyze Wikipedia articles of different cities: Boston, Seattle, San Francisco, and more.

The below code snippet downloads the relevant data into files.

```python

from pathlib import Path
import requests

wiki_titles = ["Toronto", "Seattle", "Chicago", "Boston", "Houston"]

for title in wiki_titles:
    response = requests.get(
        'https://en.wikipedia.org/w/api.php',
        params={
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            # 'exintro': True,
            'explaintext': True,
        }
    ).json()
    page = next(iter(response['query']['pages'].values()))
    wiki_text = page['extract']

    data_path = Path('data')
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", 'w') as fp:
        fp.write(wiki_text)

```

The next snippet loads all files into Document objects.

```python
# Load all wiki documents
city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(input_files=[f"data/{wiki_title}.txt"]).load_data()

```


### Defining the Set of Indexes

We will now define a set of indexes and graphs over your data. You can think of each index/graph as a lightweight structure
that solves a distinct use case.

We will first define a vector index over the documents of each city.

```python
from gpt_index import GPTSimpleVectorIndex, ServiceContext
from langchain.llms.openai import OpenAIChat

# set service context
llm_predictor_gpt4 = LLMPredictor(llm=OpenAIChat(temperature=0, model_name="gpt-4"))
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor_gpt4, chunk_size_limit=1024
)

# Build city document index
vector_indices = {}
for wiki_title in wiki_titles:
    # build vector index
    vector_indices[wiki_title] = GPTSimpleVectorIndex.from_documents(
        city_docs[wiki_title], service_context=service_context
    )
    # set id for vector index
    vector_indices[wiki_title].index_struct.index_id = wiki_title
    vector_indices[wiki_title].save_to_disk(f'index_{wiki_title}.json')
```

Querying a vector index lets us easily perform semantic search over a given city's documents.

```python
response = vector_indices["Toronto"].query("What are the sports teams in Toronto?")
print(str(response))

```

Example response:
```text
The sports teams in Toronto are the Toronto Maple Leafs (NHL), Toronto Blue Jays (MLB), Toronto Raptors (NBA), Toronto Argonauts (CFL), Toronto FC (MLS), Toronto Rock (NLL), Toronto Wolfpack (RFL), and Toronto Rush (NARL).
```

### Defining a Graph for Compare/Contrast Queries

We will now define a composed graph in order to run **compare/contrast** queries (see [use cases doc](/docs/use_cases/queries.md)).
This graph contains a keyword table composed on top of existing vector indexes. 

To do this, we first want to set the "summary text" for each vector index.

```python
index_summaries = {}
for wiki_title in wiki_titles:
    # set summary text for city
    index_summaries[wiki_title] = (
        f"This content contains Wikipedia articles about {wiki_title}. "
        f"Use this index if you need to lookup specific facts about {wiki_title}.\n"
        "Do not use this index if you want to analyze multiple cities."
    )
```

Next, we compose a keyword table on top of these vector indexes, with these indexes and summaries, in order to build the graph.


```python
from gpt_index.indices.composability import ComposableGraph

graph = ComposableGraph.from_indices(
    GPTSimpleKeywordTableIndex,
    [index for _, index in vector_indices.items()], 
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50
)

# get root index
root_index = graph.get_index(graph.index_struct.root_id, GPTSimpleKeywordTableIndex)
# set id of root index
root_index.index_struct.index_id = "compare_contrast"
root_summary = (
    "This index contains Wikipedia articles about multiple cities. "
    "Use this index if you want to compare multiple cities. "
)

```

Querying this graph (with a query transform module), allows us to easily compare/contrast between different cities. An example
is shown below - we define query_configs and send a query through this graph.

```python
# define decompose_transform
from gpt_index.indices.query.query_transform.base import DecomposeQueryTransform
decompose_transform = DecomposeQueryTransform(
    llm_predictor_chatgpt, verbose=True
)
# set query config
query_configs = [
    {
        "index_struct_type": "simple_dict",
        "query_mode": "default",
        "query_kwargs": {
            "similarity_top_k": 1
        },
        # NOTE: set query transform for subindices
        "query_transform": decompose_transform
    },
    {
        "index_struct_type": "keyword_table",
        "query_mode": "simple",
        "query_kwargs": {
            "response_mode": "tree_summarize",
            "verbose": True
        },
    },
]

# query the graph
query_str = (
    "Compare and contrast the arts and culture of Houston and Boston. "
)
response_chatgpt = graph.query(
    query_str, 
    query_configs=query_configs, 
    service_context=service_context,
)
```


### Defining the Unified Query Interface

Now that we've defined the set of indexes/graphs, we want to build an **outer abstraction** layer that provides a unified query interface
to our data structures. This means that during query-time, we can query this outer abstraction layer and trust that the right index/graph
will be used for the job. 

There are a few ways to do this, both within our framework as well as outside of it! 
- Compose a "router" on top of your existing indexes/graphs (basically expanding the graph!)
    - There are a few different "router" modules we can use, such as our tree index or vector index.
- Define each index/graph as a Tool within an agent framework (e.g. LangChain).


For the purposes of this tutorial, we follow the former approach. If you want to take a look at how the latter approach works, 
take a look at [our example tutorial here](/guides/tutorials/building_a_chatbot.md).

We define this graph using a tree index. The tree index serves as a "router". A router is at the core of defining a unified
query interface. This allows us to "route" any query to the set of indexes/graphs that you have defined under the hood.

We compose the tree index over all the vector indexes + the graph (used for compare/contrast queries).

```python
from gpt_index import GPTTreeIndex

# num children is num vector indexes + graph
num_children = len(vector_indices) + 1
outer_graph = ComposableGraph.from_indices(
    GPTTreeIndex,
    [index for _, index in vector_indices.items()] + [root_index], 
    [summary for _, summary in index_summaries.items()] + [root_summary],
    num_children=num_children
)
```


### Querying our Unified Interface

The advantage of a unified query interface is that it can now handle different types of queries.

It can now handle queries about specific cities (by routing to the specific city vector index), 
and also compare/contrast different cities.


```python
# set query config
query_configs = [
    {
        "index_struct_type": "keyword_table",
        "query_mode": "simple",
        "query_kwargs": {
            "response_mode": "tree_summarize",
            "verbose": True
        },
    },
    {
        "index_struct_type": "tree",
        "query_mode": "default",
        
    }
]
for wiki_title in wiki_titles:
    query_config = {
        "index_struct_id": wiki_title,
        "index_struct_type": "simple_dict",
        "query_mode": "default",
        "query_kwargs": {
            "similarity_top_k": 1
        },
        # NOTE: set query transform for subindices
        "query_transform": decompose_transform
    }
    query_configs.append(query_config)

```

Let's take a look at a few examples!

**Asking a Compare/Contrast Question**

```python
# ask a compare/contrast question 
response = outer_graph.query(
    "Compare and contrast the arts and culture of Houston and Boston.",
    query_configs=query_configs,
    service_context=service_context
)
print(str(response)
```


**Asking Questions about specific Cities**

```python

response = outer_graph.query("What are the sports teams in Toronto?")
print(str(response))

```

This "outer" abstraction is able to handle different queries by routing to the right underlying abstractions.










