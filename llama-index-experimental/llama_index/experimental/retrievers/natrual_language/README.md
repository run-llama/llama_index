# LlamaIndex Experimental Natual Language Retrievers

This experimental feature is enabling using natural language to retrieve information from

- Pandas dataframes
- CSV files
- JSON objects

Compare to other approaches this is using [duckDb](https://duckdb.org/) to perform KQL queries instead of python code. This is important as it addresses security concerns when running arbitrary code. The duckDb session is an in memory one and the original data cannot be altered by the retriever.

The schema is also used to generate a description of the set and what could be used for. The description and ontology are then used to calculate a ranking score against the query bundle.
