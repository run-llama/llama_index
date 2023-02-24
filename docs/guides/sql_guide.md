# A Guide to LlamaIndex + Structured Data

A lot of modern data systems depend on structured data, such as a Postgres DB or a Snowflake data warehouse.
LlamaIndex provides a lot of advanced features, powered by LLM's, to both create structured data from
unstructured data, as well as analyze this structured data through augmented text-to-SQL capabilities.

This guide helps walk through each of these capabilities. Specifically, we cover the following topics:
- **Inferring Structured Datapoints**: Converting unstructured data to structured data.
- **Text-to-SQL (basic)**: How to query a set of tables using natural language.
- **Injecting Context**: How to inject context for each table into the text-to-SQL prompt. The context
    can be manually added, or ti can be derived from unstructured documents.
- **Storing Context within an Index**: By default, we directly insert the context into the prompt. Sometimes this is not 
    feasible if the context is large. Here we show how you can actually use a LlamaIndex data structure
    to contain the table context!

We will walk through a toy example table which contains city/population/country information.

## Setup

First, we use SQLAlchemy to setup a simple sqlite db:
```python
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, select, column

engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData(bind=engine)

```

We then create a toy `city_stats` table:
```python
# create city SQL table
table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)
metadata_obj.create_all()
```

Now it's time to insert some datapoints!

If you want to look into filling into this table by inferring structured datapoints
from unstructured data, take a look at the below section. Otherwise, you can choose
to directly populate this table:

```python
from sqlalchemy import insert
rows = [
    {"city_name": "Toronto", "population": 2731571, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13929286, "country": "Japan"},
    {"city_name": "Berlin", "population": 600000, "country": "United States"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.connect() as connection:
        cursor = connection.execute(stmt)
```

Finally, we can wrap the SQLAlchemy engine with our SQLDatabase wrapper;
this allows the db to be used within LlamaIndex:

```python
from llama_index import SQLDatabase

sql_database = SQLDatabase(engine, include_tables=["city_stats"])

```

If the db is already populated with data, we can instantiate the SQL index
with a blank documents list. Otherwise see the below section.

```python
index = GPTSQLStructStoreIndex(
    [],
    sql_database=sql_database, 
    table_name="city_stats",
)
```

## Inferring Structured Datapoints

LlamaIndex offers the capability to convert unstructured datapoints to structured
data. In this section, we show how we can populate the `city_stats` table
by ingesting Wikipedia articles about each city.

First, we use the Wikipedia reader from LlamaHub to load some pages 
regarding the relevant data.

```python
from llama_index import download_loader

WikipediaReader = download_loader("WikipediaReader")
wiki_docs = WikipediaReader().load_data(pages=['Toronto', 'Berlin', 'Tokyo'])

```

When we build the SQL index, we can specify these docs as the 
first input; these documents will be converted
to structured datapoints and inserted into the db:

```python
from llama_index import GPTSQLStructStoreIndex, SQLDatabase

sql_database = SQLDatabase(engine, include_tables=["city_stats"])
# NOTE: the table_name specified here is the table that you
# want to extract into from unstructured documents.
index = GPTSQLStructStoreIndex(
    wiki_docs, 
    sql_database=sql_database, 
    table_name="city_stats",
)
```

You can take a look at the current table to verify that the datapoints have been inserted!

```python
# view current table
stmt = select(
    [column("city_name"), column("population"), column("country")]
).select_from(city_stats_table)

with engine.connect() as connection:
    results = connection.execute(stmt).fetchall()
    print(results)
```

## Text-to-SQL (basic)

LlamaIndex offers "text-to-SQL" capabilities, both at a very basic level
and also at a more advanced level. In this section, we show how to make use
of these text-to-SQL capabilities at a basic level.

A simple example is shown here:

```python
# set Logging to DEBUG for more detailed outputs
response = index.query("Which city has the highest population?", mode="default")
print(response)

```

You can access the underlying derived SQL query through `response.extra_info['sql_query']`.
It should look something like this:
```sql
SELECT city_name, population
FROM city_stats
ORDER BY population DESC
LIMIT 1
```

## Injecting Context

By default, the text-to-SQL prompt just injects the table schema information into the prompt.
However, oftentimes you may want to add your own context as well. This section shows you how
you can add context, either manually, or extracted through documents.





## Storing Context within an Index



This guide describes how each index works with diagrams. We also visually highlight our "Response Synthesis" modes.

Some terminology:
- **Node**: Corresponds to a chunk of text from a Document. GPT Index takes in Document objects and internally parses/chunks them into Node objects.
- **Response Synthesis**: Our module which synthesizes a response given the retrieved Node. You can see how to 
    [specify different response modes](setting-response-mode) here. 
    See below for an illustration of how each response mode works.

## List Index

The list index simply stores Nodes as a sequential chain.

![](/_static/indices/list.png)

### Querying

During query time, if no other query parameters are specified, GPT Index simply all Nodes in the list into
our Reponse Synthesis module.

![](/_static/indices/list_query.png)

The list index does offer numerous ways of querying a list index, from an embedding-based query which 
will fetch the top-k neighbors, or with the addition of a keyword filter, as seen below:

![](/_static/indices/list_filter_query.png)


## Vector Store Index

The vector store index stores each Node and a corresponding embedding in a [Vector Store](vector-store-index).

![](/_static/indices/vector_store.png)

### Querying

Querying a vector store index involves fetching the top-k most similar Nodes, and passing
those into our Response Synthesis module.

![](/_static/indices/vector_store_query.png)

## Tree Index

The tree index builds a hierarchical tree from a set of Nodes (which become leaf nodes in this tree).

![](/_static/indices/tree.png)

### Querying

Querying a tree index involves traversing from root nodes down 
to leaf nodes. By default, (`child_branch_factor=1`), a query
chooses one child node given a parent node. If `child_branch_factor=2`, a query
chooses two child nodes per parent.

![](/_static/indices/tree_query.png)

## Keyword Table Index

The keyword table index extracts keywords from each Node and builds a mapping from 
each keyword to the corresponding Nodes of that keyword.

![](/_static/indices/keyword.png)

### Querying

During query time, we extract relevant keywords from the query, and match those with pre-extracted
Node keywords to fetch the corresponding Nodes. The extracted Nodes are passed to our 
Response Synthesis module.

![](/_static/indices/keyword_query.png)

## Response Synthesis

GPT Index offers different methods of synthesizing a response. The way to toggle this can be found in our 
[Usage Pattern Guide](setting-response-mode). Below, we visually highlight how each response mode works.

### Create and Refine

Create and refine is an iterative way of generating a response. We first use the context in the first node, along
with the query, to generate an initial answer. We then pass this answer, the query, and the context of the second node
as input into a "refine prompt" to generate a refined answer. We refine through N-1 nodes, where N is the total 
number of nodes.

![](/_static/indices/create_and_refine.png)

### Tree Summarize

Tree summarize is another way of generating a response. We essentially build a tree index
over the set of candidate nodes, with a *summary prompt* seeded with the query. The tree
is built in a bottoms-up fashion, and in the end the root node is returned as the response.

![](/_static/indices/tree_summarize.png)