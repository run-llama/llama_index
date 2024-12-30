# LlamaIndex Readers Integration: Iceberg

```bash
pip install llama-index-readers-iceberg
```

This loader fetches data from Apache Iceberg tables.

## Usage

To use this loader, you need to pass in an Intercom account access token.

```python
from llama_index.readers.iceberg import IcebergReader

docs = IcebergReader().load_data(
    profile_name="my_profile",
    region="us-west-2",
    namespace="my_dataset",
    table="my_table",
    metadata_columns=["_id", "_age", "_name"],
)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
