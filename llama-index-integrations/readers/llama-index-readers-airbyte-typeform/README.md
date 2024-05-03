# Airbyte Typeform Loader

```bash
pip install llama-index-readers-airbyte-typeform
```

The Airbyte Typeform Loader allows you to access different Typeform objects.

## Usage

Here's an example usage of the AirbyteTypeformReader.

```python
from llama_index.readers.airbyte_typeform import AirbyteTypeformReader

typeform_config = {
    # ...
}
reader = AirbyteTypeformReader(config=typeform_config)
documents = reader.load_data(stream_name="forms")
```

## Configuration

Check out the [Airbyte documentation page](https://docs.airbyte.com/integrations/sources/typeform/) for details about how to configure the reader.
The JSON schema the config object should adhere to can be found on Github: [https://github.com/airbytehq/airbyte/blob/master/airbyte-integrations/connectors/source-typeform/source_typeform/spec.json](https://github.com/airbytehq/airbyte/blob/master/airbyte-integrations/connectors/source-typeform/source_typeform/spec.json).

The general shape looks like this:

```python
{
    "credentials": {
        "auth_type": "Private Token",
        "access_token": "<your auth token>",
    },
    "start_date": "<date from which to start retrieving records from in ISO format, e.g. 2020-10-20T00:00:00Z>",
    "form_ids": [
        "<id of form to load records for>"
    ],  # if omitted, records from all forms will be loaded
}
```

By default all fields are stored as metadata in the documents and the text is set to the JSON representation of all the fields. Construct the text of the document by passing a `record_handler` to the reader:

```python
def handle_record(record, id):
    return Document(
        doc_id=id, text=record.data["title"], extra_info=record.data
    )


reader = AirbyteTypeformReader(
    config=typeform_config, record_handler=handle_record
)
```

## Lazy loads

The `reader.load_data` endpoint will collect all documents and return them as a list. If there are a large number of documents, this can cause issues. By using `reader.lazy_load_data` instead, an iterator is returned which can be consumed document by document without the need to keep all documents in memory.

## Incremental loads

This loader supports loading data incrementally (only returning documents that weren't loaded last time or got updated in the meantime):

```python
reader = AirbyteTypeformReader(config={...})
documents = reader.load_data(stream_name="forms")
current_state = reader.last_state  # can be pickled away or stored otherwise

updated_documents = reader.load_data(
    stream_name="forms", state=current_state
)  # only loads documents that were updated since last time
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
