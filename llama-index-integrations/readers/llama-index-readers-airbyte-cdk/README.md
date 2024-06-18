# Airbyte CDK Loader

```bash
pip install llama-index-readers-airbyte-cdk
```

The Airbyte CDK Loader is a shim for sources created using the [Airbyte Python CDK](https://docs.airbyte.com/connector-development/cdk-python/). It allows you to load data from any Airbyte source into LlamaIndex.

## Installation

- Install llama-index reader: `pip install llama-index-readers-airbyte-cdk`
- Install airbyte-cdk: `pip install airbyte-cdk`
- Install a source via git (or implement your own): `pip install git+https://github.com/airbytehq/airbyte.git@master#egg=source_github&subdirectory=airbyte-integrations/connectors/source-github`

## Usage

Implement and import your own source. You can find lots of resources for how to achieve this on the [Airbyte documentation page](https://docs.airbyte.com/connector-development/).

Here's an example usage of the AirbyteCdkReader.

```python
from llama_index.readers.airbyte_cdk import AirbyteCDKReader
from source_github.source import (
    SourceGithub,
)  # this is just an example, you can use any source here - this one is loaded from the Airbyte Github repo via pip install git+https://github.com/airbytehq/airbyte.git@master#egg=source_github&subdirectory=airbyte-integrations/connectors/source-github`


github_config = {
    # ...
}
reader = AirbyteCDKReader(source_class=SourceGithub, config=github_config)
documents = reader.load_data(stream_name="issues")
```

By default all fields are stored as metadata in the documents and the text is set to the JSON representation of all the fields. Construct the text of the document by passing a `record_handler` to the reader:

```python
def handle_record(record, id):
    return Document(
        doc_id=id, text=record.data["title"], extra_info=record.data
    )


reader = AirbyteCDKReader(
    source_class=SourceGithub,
    config=github_config,
    record_handler=handle_record,
)
```

## Lazy loads

The `reader.load_data` endpoint will collect all documents and return them as a list. If there are a large number of documents, this can cause issues. By using `reader.lazy_load_data` instead, an iterator is returned which can be consumed document by document without the need to keep all documents in memory.

## Incremental loads

If a stream supports it, this loader can be used to load data incrementally (only returning documents that weren't loaded last time or got updated in the meantime):

```python
reader = AirbyteCDKReader(source_class=SourceGithub, config=github_config)
documents = reader.load_data(stream_name="issues")
current_state = reader.last_state  # can be pickled away or stored otherwise

updated_documents = reader.load_data(
    stream_name="issues", state=current_state
)  # only loads documents that were updated since last time
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
