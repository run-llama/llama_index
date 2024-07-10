# Box simple reader

This loader reads files from Box using the LLama Index Simple Reader, and does not take advantage of any Box specific features.
To use this loader, you need to pass a Box Client, a list of file id's or folder id.

## Installation

```bash
pip install llama-index-readers-box
```

## Parameters

### box_client

The Box Client represent the set for a Box API connection. It can be created using either the Client Credential Grant (CCG) or JSON Web Tokens (JWT).

It will cache the access tokens, reusing them for subsequent requests, and automatically refresh them when they expire.

### folder_id

You can extract a folder_id directly from its drive URL.

For example, the folder_id of `https://app.box.com/folder/273980493541` is `273980493541`.

### is_recursive

The reader can transverse the folder recursively to get all the files inside the folder, and its sub-folders.

> [!WARNING]
> There can be an overwhelming amount of files and folders, at which point the reader becomes impractical.

### file_ids

You can extract a file_id directly from its sharable drive URL.

For example, the file_id of `https://app.box.com/file/1584054196674` is `1584054196674`.

The reader expects a list of file ids as strings to load the files.

<!---
### query_string

You can also filter the files by the query string e.g.: `query_string="name contains 'test'"`
It gives more flexibility to filter the documents. More info: https://developers.google.com/drive/api/v3/search-files
--->

## Usage

#### With CCG authentication

```python
from box_sdk_gen import CCGConfig

ccg_conf = CCGConfig(
    client_id="your_client_id",
    client_secret="your_client_secret",
    enterprise_id="your_enterprise_id",
    user_id="your_ccg_user_id",  # optional
)
auth = BoxCCGAuth(ccg_conf)
client = BoxClient(auth)
reader = BoxReader(box_client=client)

#### Using folder id
documents = loader.load_data(folder_id="folder_id")

#### Using file ids
documents = loader.load_data(file_ids=["file_id1", "file_id2"])
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index).
