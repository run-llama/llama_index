# Airtable Loader

```bash
pip install llama-index-readers-airtable
```

This loader loads documents from Airtable. The user specifies an API token to initialize the AirtableReader. They then specify a `table_id` and a `base_id` to load in the corresponding Document objects.

## Usage

Here's an example usage of the AirtableReader.

```python
import os

from llama_index.readers.airtable import AirtableReader

reader = AirtableReader("<Airtable_TOKEN>")
documents = reader.load_data(table_id="<TABLE_ID>", base_id="<BASE_ID>")
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
