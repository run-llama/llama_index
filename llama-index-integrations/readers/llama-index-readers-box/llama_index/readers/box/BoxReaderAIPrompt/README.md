# Box AI Prompt reader

`pip install llama-index-readers-box`

This loader reads files from Box using Box AI, to extract text.
To use this loader, you need to pass a Box Client, a list of file id's or folder id.

> [!IMPORTANT]
> Box AI features are only available to E+ customers.

### ai_prompt

The AI Prompt is a tool that helps you generate text using Box AI. It can be used to generate text, answer questions, and more.

### individual_document_prompt

By default the AI Prompt will use the context of a single document `individual_document_prompt=True`, however the Box AI has the capability to answer questions looking at the context of multiple documents.

For example suppose you want to get a sentiment analysis from support requests.

You can pass a list of support requests with `individual_document_prompt=True` and the AI Prompt will generate a sentiment analysis for each one.

On the other hand if you want to get a sentiment analysis from support requests by customer, you can pass a list of support requests with `individual_document_prompt=False` and the AI Prompt will generate a sentiment analysis for all the support requests.

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

#### Using CCG authentication

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
