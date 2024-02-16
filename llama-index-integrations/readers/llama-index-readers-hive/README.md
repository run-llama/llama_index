# Hive Loader

The Hive Loader returns a set of texts corresponding to documents from Hive based on the customized query.
The user initializes the loader with Hive connection args and then using query to fetch data from Hive.

## Usage

Here's an example usage of the hiveReader to load 100 documents.

```python
from llama_index import download_loader

HiveReader = download_loader("HiveReader")

reader = HiveReader(
    host="localhost",
    port=10000,
    database="PERSON_DB",
    username="hiveuser_test",
    auth="NOSASL",
)

query = "SELECT * FROM p1 LIMIT 100"
documents = reader.load_data(query=query)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/run-llama/llama-hub/tree/main/llama_hub) for examples.
