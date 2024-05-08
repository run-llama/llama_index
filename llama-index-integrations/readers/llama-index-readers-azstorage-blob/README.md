# Azure Storage Blob Loader

```bash
pip install llama-index-readers-azstorage-blob
```

This loader parses any file stored as an Azure Storage blob or the entire container (with an optional prefix / attribute filter) if no particular file is specified. When initializing `AzStorageBlobReader`, you may pass in your account url with a SAS token or crdentials to authenticate.

All files are temporarily downloaded locally and subsequently parsed with `SimpleDirectoryReader`. Hence, you may also specify a custom `file_extractor`, relying on any of the loaders in this library (or your own)! If you need a clue on finding the file extractor object because you'd like to use your own file extractor, follow this sample.

```python
import llama_index

file_extractor = llama_index.readers.file.base.DEFAULT_FILE_READER_CLS

# Make sure to use an instantiation of a class
file_extractor.update({".pdf": SimplePDFReader()})
```

## Usage

To use this loader, you need to pass in the name of your Azure Storage Container. After that, if you want to just parse a single file, pass in its blob name. Note that if the file is nested in a subdirectory, the blob name should contain the path such as `subdirectory/input.txt`. This loader is a thin wrapper over the [Azure Blob Storage Client for Python](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?tabs=managed-identity%2Croles-azure-portal%2Csign-in-azure-cli), see [ContainerClient](https://learn.microsoft.com/en-us/python/api/azure-storage-blob/azure.storage.blob.containerclient?view=azure-python) for detailed parameter usage options.

### Using a Storage Account SAS URL

```python
from llama_index.readers.azstorage_blob import AzStorageBlobReader

loader = AzStorageBlobReader(
    container="scrabble-dictionary",
    blob="dictionary.txt",
    account_url="<SAS_URL>",
)

documents = loader.load_data()
```

### Using a Storage Account with connection string

The sample below will download all files in a container, by only specifying the storage account's connection string and the container name.

```python
from llama_index.readers.azstorage_blob import AzStorageBlobReader

loader = AzStorageBlobReader(
    container_name="<CONTAINER_NAME>",
    connection_string="<STORAGE_ACCOUNT_CONNECTION_STRING>",
)

documents = loader.load_data()
```

### Using Azure AD

Ensure the Azure Identity library is available `pip install azure-identity`

The sample below downloads all files in the container using the default credential, alternative credential options are available such as a service principal `ClientSecretCredential`

```python
from azure.identity import DefaultAzureCredential

default_credential = DefaultAzureCredential()

from llama_index.readers.azstorage_blob import AzStorageBlobReader

loader = AzStorageBlobReader(
    container_name="scrabble-dictionary",
    account_url="https://<storage account name>.blob.core.windows.net",
    credential=default_credential,
)

documents = loader.load_data()
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).

### Updates

#### [2023-12-14] by [JAlexMcGraw](https://github.com/JAlexMcGraw) (#765)

- Added functionality to allow user to connect to blob storage with connection string
- Changed temporary file names from random to back to original names
