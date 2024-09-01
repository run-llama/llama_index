# LlamaIndex Docstore Integration: Azure Table Storage

`AzureDocumentStore` allows you to use any compatible Azure Table Storage or CosmosDB as a document store for LlamaIndex.

## Installation

Before using the `AzureDocumentStore`, ensure you have Python installed and then proceed to install the required packages:

```bash
pip install llama-index-storage-docstore-azure
pip install azure-data-tables
pip install azure-identity  # Only needed for AAD token authentication
```

## Initializing `AzureDocumentStore`

`AzureDocumentStore` can be initialized in several ways depending on the authentication method and the Azure service (Table Storage or Cosmos DB) you are using:

### 1. Using a Connection String

```py
from llama_index.storage.docstore.azure import AzureDocumentStore
from llama_index.storage.kvstore.azure.base import ServiceMode

store = AzureDocumentStore.from_connection_string(
    connection_string="your_connection_string_here",
    namespace="your_namespace",
    service_mode=ServiceMode.STORAGE,  # or ServiceMode.COSMOS
)
```

### 2. Using Account Name and Key

```py
store = AzureDocumentStore.from_account_and_key(
    account_name="your_account_name",
    account_key="your_account_key",
    service_mode=ServiceMode.STORAGE,  # or ServiceMode.COSMOS
)
```

### 3. Using SAS Token

```py
store = AzureDocumentStore.from_sas_token(
    endpoint="your_endpoint",
    sas_token="your_sas_token",
    service_mode=ServiceMode.STORAGE,  # or ServiceMode.COSMOS
)
```

### 4. Using Azure Active Directory (AAD) Token

```py
store = AzureDocumentStore.from_aad_token(
    endpoint="your_endpoint",
    service_mode=ServiceMode.STORAGE,  # or ServiceMode.COSMOS
)
```

## End-to-end example:

```py
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex

reader = SimpleDirectoryReader("./data/paul_graham/")
documents = reader.load_data()
nodes = SentenceSplitter().get_nodes_from_documents(documents)

storage_context = StorageContext.from_defaults(
    docstore=AzureDocumentStore.from_account_and_key(
        "your_account_name",
        "your_account_key",
        service_mode=ServiceMode.STORAGE,
    ),
)

vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

storage_context.docstore.add_documents(nodes)
```
