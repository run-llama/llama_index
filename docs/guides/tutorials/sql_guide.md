# A Guide to LlamaIndex + Structured Data

A lot of modern data systems depend on structured data, such as a Postgres DB or a Snowflake data warehouse.
LlamaIndex provides a lot of advanced features, powered by LLM's, to both create structured data from
unstructured data, as well as analyze this structured data through augmented text-to-SQL capabilities.

This guide helps walk through each of these capabilities. Specifically, we cover the following topics:
- **Setup**: Defining up our example SQL Table.
- **Building our Table Index**: How to go from sql database to a Table Schema Index
- **Using natural language SQL queries**: How to query our SQL database using natural language.

We will walk through a toy example table which contains city/population/country information.
A notebook for this tutorial is [available here](../../examples/index_structs/struct_indices/SQLIndexDemo.ipynb).
## Setup

First, we use SQLAlchemy to setup a simple sqlite db:
```python
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, select, column

engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()
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
metadata_obj.create_all(engine)
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

## Building our Table Index
Now that we have our SQLDatabase object, we would like to store the table schema
in an index so that during query time we can retrieve the right schema.

The way we can do this is using the SQLTableNodeMapping object, which takes in a 
SQLDatabase and produces a Node object for each SQLTableSchema object passed into the
ObjectIndex constructor.

```python
table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [(SQLTableSchema(table_name="city_stats"))]

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
```
Here you can see we define our table_node_mapping, and a single SQLTableSchema with the
"city_stats" table name. We pass these into the ObjectIndex constructor, along with the
VectorStoreIndex class definition we want to use. This will give us a VectorStoreIndex where
each Node contains table schema and other context information. You can also add any additional
context information you'd like.
```python
# manually set context text
city_stats_text = (
    "This table gives information regarding the population and country of a given city.\n"
    "The user will query with codewords, where 'foo' corresponds to population and 'bar'"
    "corresponds to city."
)

table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [(SQLTableSchema(table_name="city_stats", context_str=city_stats_text))]
```

## Using natural language SQL queries
Once we have defined our table schema index obj_index, we can construct a SQLTableRetrieverQueryEngine
by passing in our SQLDatabase, and a retriever constructed from our object index.

```python
query_engine = SQLTableRetrieverQueryEngine(
    sql_database, obj_index.as_retriever(similarity_top_k=1)
)
response = query_engine.query("Which city has the highest population?")
print(response)
```

## Concluding Thoughts

This is it for now! We're constantly looking for ways to improve our structured data support.
If you have any questions let us know in [our Discord](https://discord.gg/dGcwcsnxhU).


