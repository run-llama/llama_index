# Hive Loader

> [!NOTE] > _This package has been deprecated starting from `v0.3.1`_

```bash
pip install llama-index-readers-hive
```

The Hive Loader returns a set of texts corresponding to documents from Hive based on the customized query.
The user initializes the loader with Hive connection args and then using query to fetch data from Hive.

## Usage

Here's an example usage of the hiveReader to load 100 documents.

```python
from llama_index.readers.hive import HiveReader

reader = HiveReader(
    host="localhost",
    port=10000,
    database="PERSON_DB",
    username="hiveuser_test",
    auth="NOSASL",
)

query = "SELECT * FROM p1 LIMIT 100"
documents = reader.load_data(query=query)
print(documents)
# test with params
query = "SELECT * FROM p1 WHERE id = %s"
params = ("1",)
documents1 = reader.load_data(query=query, params=params)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
