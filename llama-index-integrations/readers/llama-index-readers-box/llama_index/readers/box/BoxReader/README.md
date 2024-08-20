# Box Reader

This loader reads files from Box using the LLama Index Simple Reader, and does not take advantage of any Box specific features.

## Usage

### Instantiate the reader

To instantiate the reader you only need a `BoxClient` object.

```python
# Using CCG authentication

from llama_index.readers.box import BoxReader
from box_sdk_gen import CCGConfig, BoxCCGAuth, BoxClient

ccg_conf = CCGConfig(
    client_id="your_client_id",
    client_secret="your_client_secret",
    enterprise_id="your_enterprise_id",
    user_id="your_ccg_user_id",  # optional
)
auth = BoxCCGAuth(ccg_conf)
client = BoxClient(auth)

reader = BoxReader(box_client=client)
```

### Load data

This method retrieves Box files based on the provided parameters and
processes them into a structured format using a SimpleDirectoryReader.

#### Args:

- **`folder_id (Optional[str])`**: The ID of the Box folder to load
  data from. If provided, along with is_recursive set to True, retrieves
  data from sub-folders as well. Defaults to None.
- **`file_ids (Optional[List[str]])`**: A list of Box file IDs to
  load data from. If provided, folder_id is ignored. Defaults to None.
- **`is_recursive (bool = False)`**: If True and folder_id is provided, retrieves
  data from sub-folders within the specified folder. Defaults to False.

> [!WARNING]
> There can be an overwhelming amount of files and folders, at which point the reader becomes impractical.

#### Returns:

- **`List[Document]`**: A list of Document objects containing the processed data
  extracted from the Box files.

#### Raises:

- **`BoxAPIError`**: If an error occurs while interacting with the Box API.

#### Example

```python
# Using a list of file ids
docs = reader.load_data(file_ids=["test_csv_id"])

# Using a folder id
docs = reader.load_data(folder_id="test_folder_id")
```

### Load resource

Load data from a specific resource.

#### Args:

- **`resource (str)`**: The resource identifier (`file_id`).

#### Returns:

- **`List[Document]`**: A list of documents loaded from the resource.

#### Example

```python
doc = reader.load_resource("test_csv_id")
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
