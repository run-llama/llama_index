# Airbyte Gong Loader

```bash
pip install llama-index-readers-airbyte-gong
```

The Airbyte Gong Loader allows you to access different Gong objects.

## Usage

Here's an example usage of the AirbyteGongReader.

```python
from llama_index.readers.airbyte_gong import AirbyteGongReader

gong_config = {
    # ...
}
reader = AirbyteGongReader(config=gong_config)
documents = reader.load_data(stream_name="calls")
```

## Configuration

Check out the [Airbyte documentation page](https://docs.airbyte.com/integrations/sources/gong/) for details about how to configure the reader.
The JSON schema the config object should adhere to can be found on Github: [https://github.com/airbytehq/airbyte/blob/master/airbyte-integrations/connectors/source-gong/source_gong/spec.yaml](https://github.com/airbytehq/airbyte/blob/master/airbyte-integrations/connectors/source-gong/source_gong/spec.yaml).

The general shape looks like this:

```python
{
    "access_key": "<access key name>",
    "access_key_secret": "<access key secret>",
    "start_date": "<date from which to start retrieving records from in ISO format, e.g. 2020-10-20T00:00:00Z>",
}
```

By default all fields are stored as metadata in the documents and the text is set to the JSON representation of all the fields. Construct the text of the document by passing a `record_handler` to the reader:

```python
def handle_record(record, id):
    return Document(
        doc_id=id, text=record.data["title"], extra_info=record.data
    )


reader = AirbyteGongReader(config=gong_config, record_handler=handle_record)
```

## Lazy loads

The `reader.load_data` endpoint will collect all documents and return them as a list. If there are a large number of documents, this can cause issues. By using `reader.lazy_load_data` instead, an iterator is returned which can be consumed document by document without the need to keep all documents in memory.

## Incremental loads

This loader supports loading data incrementally (only returning documents that weren't loaded last time or got updated in the meantime):

```python
reader = AirbyteGongReader(config={...})
documents = reader.load_data(stream_name="calls")
current_state = reader.last_state  # can be pickled away or stored otherwise

updated_documents = reader.load_data(
    stream_name="calls", state=current_state
)  # only loads documents that were updated since last time
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
