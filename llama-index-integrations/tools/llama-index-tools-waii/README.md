# Waii Tool

This tool connects to database connections managed by Waii, which allows generic SQL queries, do performance analyze, describe a SQL query, and more.

## Usage

First you need to create a waii.ai account, you request an account from [here](https://waii.ai/).

Initialize the tool with your account credentials:

```python
from llama_index.tools.waii import WaiiToolSpec

waii_tool = WaiiToolSpec(
    url="https://tweakit.waii.ai/api/",
    # API Key of Waii (not OpenAI API key)
    api_key="...",
    # Connection key of WAII connected database, see https://github.com/waii-ai/waii-sdk-py#get-connections
    database_key="...",
)
```

## Tools

The tools available are:

- `get_answer`: Get answer to natural language question (which generate a SQL query, run it, explain the result)
- `describe_query`: Describe a SQL query
- `performance_analyze`: Analyze performance of a SQL query (by query_id)
- `diff_query`: Compare two SQL queries
- `describe_dataset`: Describe dataset, such as table, schema, etc.
- `transcode`: Transcode SQL query to another SQL dialect
- `get_semantic_contexts`: Get semantic contexts of a SQL query
- `generate_query_only`: Generate SQL query only (not run it)
- `run_query`: Run a SQL query

You can also load the data directly call `load_data`

## Examples

### Load data

```python
documents = waii_tool.load_data("Get all tables with their number of columns")
index = VectorStoreIndex.from_documents(documents).as_query_engine()

print(index.query("Which table contains most columns?"))
```

### Use as a Tool

#### Initialize the agent:

```python
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

agent = OpenAIAgent.from_tools(
    waii_tool.to_tool_list(), llm=OpenAI(model="gpt-4-1106-preview")
)
```

#### Ask simple question

```python
agent.chat("Give me top 3 countries with the most number of car factory")
agent.chat("What are the car factories of these countries")
```

#### Do performance analyze

```python
agent.chat("Give me top 3 longest running queries, and their duration.")
agent.chat("analyze the 2nd-longest running query")
```

#### Diff two queries

```python
previous_query = """
SELECT
    employee_id,
    department,
    salary,
    AVG(salary) OVER (PARTITION BY department) AS department_avg_salary,
    salary - AVG(salary) OVER (PARTITION BY department) AS diff_from_avg
FROM
    employees;
"""
current_query = """
SELECT
    employee_id,
    department,
    salary,
    MAX(salary) OVER (PARTITION BY department) AS department_max_salary,
    salary - AVG(salary) OVER (PARTITION BY department) AS diff_from_avg
FROM
    employees;
LIMIT 100;
"""
agent.chat(f"tell me difference between {previous_query} and {current_query}")
```

#### Describe dataset

```python
agent.chat("Summarize the dataset")
agent.chat("Give me questions which I can ask about this dataset")
```

#### Describe a query

```python
q = """
SELECT
    employee_id,
    department,
    salary,
    AVG(salary) OVER (PARTITION BY department) AS department_avg_salary,
    salary - AVG(salary) OVER (PARTITION BY department) AS diff_from_avg
FROM
    employees;
"""
agent.chat(f"what this query can do? {q}")
```

#### Migrate query to another dialect

```python
q = """
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col
from pyspark.sql.window import Window

# Initialize Spark session
spark = SparkSession.builder.appName("example").getOrCreate()

# Assuming you have a DataFrame called 'employees'
# If not, you need to read your data into a DataFrame first

# Define window specification
windowSpec = Window.partitionBy("department")

# Perform the query
result = (employees
          .select(
              col("employee_id"),
              col("department"),
              col("salary"),
              avg("salary").over(windowSpec).alias("department_avg_salary"),
              (col("salary") - avg("salary").over(windowSpec)).alias("diff_from_avg")
          ))

# Show the result
result.show()
"""
agent.chat(f"translate this pyspark query {q}, to Snowflake")
```

### Use Waii API directly

You can also use Waii API directly, see [here](https://github.com/waii-ai/waii-sdk-py)
