# Box Text Extraction

A reader class for loading text content from Box files.

This class inherits from the `BoxReaderBase` class and specializes in
extracting plain text content from Box files. It utilizes the provided
BoxClient object to interact with the Box API and retrieves the text
representation of the files.

> [!TIP]
> For more information check the [Box text representation documentation](https://developer.box.com/guides/representations/text/).

## Usage

### Instantiate the reader

To instantiate the reader you only need a `BoxClient` object.

```python
# Using CCG authentication

from llama_index.readers.box import BoxReaderTextExtraction
from box_sdk_gen import CCGConfig, BoxCCGAuth, BoxClient

ccg_conf = CCGConfig(
    client_id="your_client_id",
    client_secret="your_client_secret",
    enterprise_id="your_enterprise_id",
    user_id="your_ccg_user_id",  # optional
)
auth = BoxCCGAuth(ccg_conf)
client = BoxClient(auth)

reader = BoxReaderTextExtraction(box_client=client)
```

### Load data

Extracts text content from Box files and creates Document objects.

This method utilizes the Box API to retrieve the text representation
(if available) of the specified Box files. It then creates Document
objects containing the extracted text and file metadata.

#### Args:

- **`file_ids (Optional[List[str]], optional)`**: A list of Box file IDs
  to extract text from. If provided, folder_id is ignored.
  Defaults to None.
- **`folder_id (Optional[str], optional)`**: The ID of the Box folder to
  extract text from. If provided, along with is_recursive set to
  True, retrieves data from sub-folders as well. Defaults to None.
- **`is_recursive (bool, optional)`**: If True and folder_id is provided,
  extracts text from sub-folders within the specified folder.
  Defaults to False.

> [!WARNING]
> There can be an overwhelming amount of files and folders, at which point the reader becomes impractical.

#### Returns:

- **`List[Document]`**: A list of Document objects containing the extracted
  text content and file metadata.

#### Example

```python
#### Using folder id
documents = reader.load_data(folder_id="folder_id")

#### Using file ids
documents = reader.load_data(file_ids=["file_id1", "file_id2"])
```

### Load resource

Load data from a specific resource.

#### Args:

- **`resource (str)`**: The resource identifier.
- **`ai_prompt (str)`**: The custom AI prompt that specifies what data to
  extract from the files.

#### Returns:

- **`List[Document]`**: A list of documents loaded from the resource.

#### Example

```python
resource_id = test_data["test_txt_invoice_id"]
docs = reader.load_resource(resource_id, ai_prompt="summarize this document")
```

### List resources

Lists the IDs of Box files based on the specified folder or file IDs.

This method retrieves a list of Box file identifiers based on the provided
parameters. You can either specify a list of file IDs or a folder ID with an
optional `is_recursive` flag to include files from sub-folders as well.

#### Args:

- **`folder_id (Optional[str])`**: The ID of the Box folder to list files
  from. If provided, along with `is_recursive` set to True, retrieves data
  from sub-folders as well. Defaults to None.
- **`file_ids (Optional[List[str]])`**: A list of Box file IDs to retrieve.
  If provided, this takes precedence over `folder_id`. Defaults to None.
- **`is_recursive (bool=False)`**: If True and `folder_id` is provided, retrieves
  resource IDs from sub-folders within the specified folder. Defaults to False.

> [!WARNING]
> There can be an overwhelming amount of files and folders, at which point the reader becomes impractical.

#### Returns:

    List[str]: A list containing the IDs of the retrieved Box files.

#### Example

```python
resources = reader.list_resources(file_ids=["test_csv_id"])
```

### Read file content

Returns the binary content of a file.

```python
input_file: Path = Path("test_csv_id")
content = reader.read_file_content(input_file)
```

### Search resources

Searches for Box resources based on specified criteria and returns a list of their IDs.

This method utilizes the Box API search functionality to find resources
matching the provided parameters. It then returns a list containing the IDs
of the found resources.

> [!TIP]
> Check out the [Box Search](https://developer.box.com/guides/search/) for more information on how to operate search.

```python
query = "invoice"
resources = reader.search_resources(query=query)
```

### Search resources by metadata

Searches for Box resources based on metadata and returns a list of their IDs.

This method utilizes the Box API search functionality to find resources
matching the provided metadata query. It then returns a list containing the IDs
of the found resources.

> [!TIP]
> Check out the [Box Metadata Query Language](https://developer.box.com/guides/metadata/queries/syntax/) for more information on how to construct queries.

```python
from_ = (
    test_data["enterprise_1234"]  # your enterprise id
    + "rbInvoicePO"  # your metadata template key
)
ancestor_folder_id = "test_folder_invoice_po_id"
query = "documentType = :docType "
query_params = {"docType": "Invoice"}

resources = reader.search_resources_by_metadata(
    from_=from_,
    ancestor_folder_id=ancestor_folder_id,
    query=query,
    query_params=query_params,
)
```

### Get resource info

Get information about a specific resource.

```python
resource = reader.get_resource_info(file_id=test_data["test_csv_id"])
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index).

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

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index).
