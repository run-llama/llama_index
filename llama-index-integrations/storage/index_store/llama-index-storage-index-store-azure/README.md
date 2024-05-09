# LlamaIndex Index_Store Integration: Azure Table Storage

`AzureIndexStore` utilizes Azure Table Storage and Cosmos DB to provide an index storage solution for indexing documents or data.

## Installation

Before using the `AzureIndexStore`, ensure you have Python installed and then proceed to install the required packages:

```bash
pip install llama-index-storage-index-store-azure
pip install azure-data-tables
pip install azure-identity  # Only needed for AAD token authentication
```

## Initializing `AzureIndexStore`

`AzureIndexStore` can be initialized in several ways depending on the authentication method and the Azure service (Table Storage or Cosmos DB) you are using:

### 1. Using a Connection String

```py
from llama_index.storage.index_store.azure import AzureIndexStore
from llama_index.storage.kvstore.azure.base import ServiceMode

store = AzureIndexStore.from_connection_string(
    connection_string="your_connection_string_here",
    namespace="your_namespace",
    service_mode=ServiceMode.STORAGE,  # or ServiceMode.COSMOS
)
```

### 2. Using Account Name and Key

```py
store = AzureIndexStore.from_account_and_key(
    account_name="your_account_name",
    account_key="your_account_key",
    service_mode=ServiceMode.STORAGE,  # or ServiceMode.COSMOS
)
```

### 3. Using SAS Token

```py
store = AzureIndexStore.from_sas_token(
    endpoint="your_endpoint",
    sas_token="your_sas_token",
    service_mode=ServiceMode.STORAGE,  # or ServiceMode.COSMOS
)
```

### 4. Using Azure Active Directory (AAD) Token

```py
store = AzureIndexStore.from_aad_token(
    endpoint="your_endpoint",
    service_mode=ServiceMode.STORAGE,  # or ServiceMode.COSMOS
)
```

## End-to-end example:

```py
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex

reader = SimpleDirectoryReader("./data/paul_graham/")
documents = reader.load_data()
nodes = SentenceSplitter().get_nodes_from_documents(documents)

storage_context = StorageContext.from_defaults(
    index_store=AzureIndexStore.from_account_and_key(
        "your_account_name",
        "your_account_key",
        service_mode=ServiceMode.STORAGE,
    ),
)

storage_context.docstore.add_documents(nodes)

keyword_table_index = SimpleKeywordTableIndex(
    nodes, storage_context=storage_context
)

vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
```
