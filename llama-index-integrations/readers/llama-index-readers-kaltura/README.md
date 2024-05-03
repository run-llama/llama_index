# Kaltura eSearch Loader

```bash
pip install llama-index-readers-kaltura-esearch
```

This loader reads Kaltura Entries from [Kaltura](https://corp.kaltura.com) based on a Kaltura eSearch API call.
Search queries can be passed as a pre-defined object of KalturaESearchEntryParams, or through a simple free text query.
The result is a list of documents containing the Kaltura Entries and Captions json.

## Parameters

### `KalturaESearchEntryParams`

This is a Kaltura class used for performing search operations in Kaltura. You can use this class to define various search criteria, such as search phrases, operators, and objects to be searched.

For example, you can search for entries with specific tags, created within a specific time frame, or containing specific metadata.

### Kaltura Configuration

To use the Kaltura eSearch Loader, you need to provide the following configuration credentials:

| Parameter          | Description                                                                                    | Default Value                                                      |
| ------------------ | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| partnerId          | Your Kaltura partner ID.                                                                       | Mandatory (no default)                                             |
| apiSecret          | Your Kaltura API secret key (aka Admin Secret).                                                | Mandatory (no default)                                             |
| userId             | Your Kaltura user ID.                                                                          | Mandatory (no default)                                             |
| ksType             | The Kaltura session type.                                                                      | KalturaSessionType.ADMIN                                           |
| ksExpiry           | The Kaltura session expiry time.                                                               | 86400 seconds                                                      |
| ksPrivileges       | The Kaltura session privileges.                                                                | "disableentitlement"                                               |
| kalturaApiEndpoint | The Kaltura API endpoint URL.                                                                  | "[https://cdnapi-ev.kaltura.com/](https://cdnapi-ev.kaltura.com/)" |
| requestTimeout     | The request timeout duration in seconds.                                                       | 500 seconds                                                        |
| shouldLogApiCalls  | If passed True, all the Kaltura API calls will also be printed to log (only use during debug). | False                                                              |

### load_data

This method run the search in Kaltura and load Kaltura entries in a list of dictionaries.

#### Method inputs

- search_params: search parameters of type KalturaESearchEntryParams with pre-set search queries. If not provided, the other parameters will be used to construct the search query.
- search_operator_and: if True, the constructed search query will have AND operator between query filters, if False, the operator will be OR.
- free_text: if provided, will be used as the free text query of the search in Kaltura.
- category_ids: if provided, will only search for entries that are found inside these category ids.
- withCaptions: determines whether or not to also download captions/transcript contents from Kaltura.
- maxEntries: sets the maximum number of entries to pull from Kaltura, between 0 to 500 (max pageSize in Kaltura).

#### Method output

Each dictionary in the response represents a Kaltura media entry, where the keys are strings (field names) and the values can be of any type:

| Column Name          | Data Type | Description                                               |
| -------------------- | --------- | --------------------------------------------------------- |
| entry_id             | str       | Unique identifier of the entry                            |
| entry_name           | str       | Name of the entry                                         |
| entry_description    | str       | Description of the entry                                  |
| entry_captions       | JSON      | Captions of the entry                                     |
| entry_media_type     | int       | Type of the media (KalturaMediaType)                      |
| entry_media_date     | int       | Date of the media Unix timestamp                          |
| entry_ms_duration    | int       | Duration of the entry in ms                               |
| entry_last_played_at | int       | Last played date of the entry Unix timestamp              |
| entry_application    | str       | The app that created this entry (KalturaEntryApplication) |
| entry_tags           | str       | Tags of the entry (comma separated)                       |
| entry_reference_id   | str       | Reference ID of the entry                                 |

## Usage

First, instantiate the KalturaReader (aka Kaltura Loader) with your Kaltura configuration credentials:

```python
from llama_index.readers.kaltura_esearch import KalturaESearchReader

loader = KalturaESearchReader(
    partnerId="INSERT_YOUR_PARTNER_ID",
    apiSecret="INSERT_YOUR_ADMIN_SECRET",
    userId="INSERT_YOUR_USER_ID",
)
```

### Using an instance of KalturaESearchEntryParams

Then, create an instance of `KalturaESearchEntryParams` and set your desired search parameters:

```python
from KalturaClient.Plugins.ElasticSearch import (
    KalturaESearchEntryParams,
    KalturaESearchEntryOperator,
    KalturaESearchOperatorType,
    KalturaESearchUnifiedItem,
)

# instantiate the params object
search_params = KalturaESearchEntryParams()

# define search parameters (for example, search for entries with a certain tag)
search_params.searchOperator = KalturaESearchEntryOperator()
search_params.searchOperator.operator = KalturaESearchOperatorType.AND_OP
search_params.searchOperator.searchItems = [KalturaESearchUnifiedItem()]
search_params.searchOperator.searchItems[0].searchTerm = "my_tag"
```

Once you have your `KalturaESearchEntryParams` ready, you can pass it to the Kaltura Loader:

```python
# Using search params
entry_docs = loader.load_data(search_params)
```

### Using Free Text Search

```python
# Simple pass the search params into the load_data method without setting search_params
entry_docs = loader.load_data(
    search_operator_and=True,
    free_text="education",
    category_ids=None,
    with_captions=True,
    max_entries=5,
)
```

For a more elaborate example, see: [llamaindex_kaltura_esearch_reader_example.py](https://gist.github.com/zoharbabin/07febcfe52b64116c9e3ba1a392b59a0)

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).

## About Kaltura

Kaltura Video Cloud is a Digital Experience Platform enabling streamlined creation, management, and distribution of media content (video, audio, image, doc, live stream, real-time video). It powers many applications across industries with collaboration, interactivity, virtual events, and deep video analytics capabilities.
