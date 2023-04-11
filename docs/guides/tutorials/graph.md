# A Guide to Creating a Unified Query Framework over your Indexes

LlamaIndex offers a variety of different [query use cases](/docs/use_cases/queries.md). 

For simple queries, we may want to use a single index data structure, such as a `GPTSimpleVectorIndex` for semantic search, or `GPTListIndex` for summarization.

For more complex queries, we may want to use a composable graph. 

But how do we integration indexes and graphs into our LLM application? Different indexes and graphs may be better suited for different types of queries that you may want to run. 

In this guide, we show how you can unify the diverse use cases of different index/graph structures under a **single** query framework.

### Setup

In this example, we will analyze Wikipedia articles of different cities: Boston, Seattle, San Francisco, and more.

The below code snippet downloads the relevant data into files.

```python
```


### Defining the Set of Indexes/Graphs

We will now define a set of indexes and graphs over your data. You can think of each index/graph as a lightweight structure
that solves a distinct use case.

We will first define a vector index and list index over the documents of each city.

Querying a vector index lets us easily perform semantic search over a given city's documents.

<example>


We will now define a composed graph in order to run **compare/contrast** queries (see [use cases doc](/docs/use_cases/queries.md)).
This graph contains a keyword table composed on top of existing vector indexes. 

Querying this graph (with a query transform module), allows us to easily compare/contrast between different cities:

<example>


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


#### Setting Summary Description for each Index
We first set a "summary description" of each index and the "top-level" index corresponding to each graph structure.

#### Composing an outer "Router" Index
Then, we compose a vector index on top of these indexes! This allows us to treat each index as a sub-node.

This outer vector index acts as a router. When a query hits this outer vector index, we will lookup the most relevant top-1 node by embedding similarity. Since this node corresponds to a subindex, we effectively then "route" the query to the subindex.

**NOTE**: You may also choose to use a tree index. Then instead of choosing child nodes based on embedding similarity, we will
choose the child node through LLM calls.

#### Querying this outer Router Index










