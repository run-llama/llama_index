# A Guide to LlamaIndex + Structured Data

A lot of modern data systems depend on structured data, such as a Postgres DB or a Snowflake data warehouse.
LlamaIndex provides a lot of advanced features, powered by LLM's, to both create structured data from
unstructured data, as well as analyze this structured data through augmented text-to-SQL capabilities.

This guide helps walk through each of these capabilities. Specifically, we cover the following topics:
- **Inferring Structured Datapoints**: Converting unstructured data to structured data.
- **Text-to-SQL (basic)**: How to query a set of tables using natural language.
- **Injecting Context**: How to inject context for each table into the text-to-SQL prompt. The context
    can be manually added, or it can be derived from unstructured documents.
- **Storing Table Context within an Index**: By default, we directly insert the context into the prompt. Sometimes this is not 
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
    {"city_name": "Berlin", "population": 600000, "country": "Germany"},
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
index = GPTSQLStructStoreIndex.from_documents(
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
query_engine = index.as_query_engine()
response = query_engine.query("Which city has the highest population?")
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

We offer you a context builder class to better manage the context within your SQL tables:
`SQLContextContainerBuilder`. This class takes in the `SQLDatabase` object,
and a few other optional parameters, and builds a `SQLContextContainer` object
that you can then pass to the index during construction + query-time.

You can add context manually to the context builder. The code snippet below shows you how:

```python
# manually set text
city_stats_text = (
    "This table gives information regarding the population and country of a given city.\n"
    "The user will query with codewords, where 'foo' corresponds to population and 'bar'"
    "corresponds to city."
)
table_context_dict={"city_stats": city_stats_text}
context_builder = SQLContextContainerBuilder(sql_database, context_dict=table_context_dict)
context_container = context_builder.build_context_container()

# building the index
index = GPTSQLStructStoreIndex.from_documents(
    wiki_docs, 
    sql_database=sql_database, 
    table_name="city_stats",
    sql_context_container=context_container
)
```

You can also choose to **extract** context from a set of unstructured Documents.
To do this, you can call `SQLContextContainerBuilder.from_documents`.
We use the `TableContextPrompt` and the `RefineTableContextPrompt` (see
the [reference docs](/reference/prompts.rst)).

```python
# this is a dummy document that we will extract context from
# in GPTSQLContextContainerBuilder
city_stats_text = (
    "This table gives information regarding the population and country of a given city.\n"
)
context_documents_dict = {"city_stats": [Document(city_stats_text)]}
context_builder = SQLContextContainerBuilder.from_documents(
    context_documents_dict, 
    sql_database
)
context_container = context_builder.build_context_container()

# building the index
index = GPTSQLStructStoreIndex.from_documents(
    wiki_docs, 
    sql_database=sql_database, 
    table_name="city_stats",
    sql_context_container=context_container,
)
```

## Storing Table Context within an Index

A database collection can have many tables, and if each table has many columns + a description associated with it,
then the total context can be quite large.

Luckily, you can choose to use a LlamaIndex data structure to store this table context! 
Then when the SQL index is queried, we can use this "side" index to retrieve the proper context
that can be fed into the text-to-SQL prompt.

Here we make use of the `derive_index_from_context` function within `SQLContextContainerBuilder`
to create a new index. You have flexibility in choosing which index class to specify + 
which arguments to pass in. We then use a helper method called `query_index_for_context`
which is a simple wrapper on the `query` call that wraps a query template + 
stores the context on the generated context container.

You can then build the context container, and pass it to the index during query-time!

```python
from llama_index import GPTSQLStructStoreIndex, SQLDatabase, GPTVectorStoreIndex
from llama_index.indices.struct_store import SQLContextContainerBuilder

sql_database = SQLDatabase(engine)
# build a vector index from the table schema information
context_builder = SQLContextContainerBuilder(sql_database)
table_schema_index = context_builder.derive_index_from_context(
    GPTVectorStoreIndex,
    store_index=True
)

query_str = "Which city has the highest population?"

# query the table schema index using the helper method
# to retrieve table context
context_builder.query_index_for_context(
    table_schema_index,
    query_str,
    store_context_str=True
)

context_container = context_builder.build_context_container()

index = GPTSQLStructStoreIndex(
    [],
    sql_database=sql_database,
    sql_context_container=context_container
)

# query the SQL index with the table context
query_engine = index.as_query_engine()
response = query_engine.query(query_str)
print(response)

```


## Concluding Thoughts

This is it for now! We're constantly looking for ways to improve our structured data support.
If you have any questions let us know in [our Discord](https://discord.gg/dGcwcsnxhU).


