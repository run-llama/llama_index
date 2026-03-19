# LlamaIndex Retrievers Integration: Snowflake Cortex Search

Retrieve documents from [Snowflake Cortex Search Service](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview) for use in LlamaIndex RAG pipelines.

Cortex Search is a fully managed search service that provides low-latency, high-quality fuzzy and semantic search over your Snowflake data.

## Installation

```bash
pip install llama-index-retrievers-cortex-search
```

## Usage

```python
from llama_index.retrievers.cortex_search import CortexSearchRetriever

retriever = CortexSearchRetriever(
    service_name="my_search_service",
    database="MY_DB",
    schema="MY_SCHEMA",
    search_column="content",  # column to use as node text
    columns=["content", "title", "url"],  # columns to return
    limit=10,
    account="ORG_ID-ACCOUNT_ID",
    user="MY_USER",
    private_key_file="/path/to/rsa_key.p8",
)

# Simple retrieval
nodes = retriever.retrieve("What is Snowflake Cortex?")
for node in nodes:
    print(f"{node.score:.3f} | {node.text[:80]}")

# With filters
retriever_filtered = CortexSearchRetriever(
    service_name="my_search_service",
    database="MY_DB",
    schema="MY_SCHEMA",
    search_column="content",
    columns=["content", "category"],
    filter_spec={"@eq": {"category": "documentation"}},
    account="ORG_ID-ACCOUNT_ID",
    user="MY_USER",
    private_key_file="/path/to/rsa_key.p8",
)

# Use as a query engine component
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.cortex import Cortex

query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    llm=Cortex(model="mistral-large2", ...),
)
response = query_engine.query("Explain Cortex Search")
```

## Authentication

Same authentication methods as `llama-index-llms-cortex`:

1. **Key-pair auth**: `private_key_file`, `account`, `user`
2. **JWT token**: `jwt_token` (string or filepath)
3. **Snowpark Session**: `session` object
4. **Environment variable**: `SNOWFLAKE_KEY_FILE`
5. **SPCS auto-detect**: Default OAuth token in Snowpark Container Services

## Filter Operators

Cortex Search supports these filter operators in `filter_spec`:

| Operator    | Description           | Example                                       |
| ----------- | --------------------- | --------------------------------------------- |
| `@eq`       | Exact match           | `{"@eq": {"category": "docs"}}`               |
| `@contains` | Contains value        | `{"@contains": {"tags": "python"}}`           |
| `@gte`      | Greater than or equal | `{"@gte": {"date": "2024-01-01"}}`            |
| `@lte`      | Less than or equal    | `{"@lte": {"price": 100}}`                    |
| `@and`      | Logical AND           | `{"@and": [{"@eq": {...}}, {"@gte": {...}}]}` |
| `@or`       | Logical OR            | `{"@or": [{"@eq": {...}}, {"@eq": {...}}]}`   |
