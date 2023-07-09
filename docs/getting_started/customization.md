# Customization

## Service Modules
**Configuring document chunk size**
```python
from llama_index import ServiceContext

service_context = ServiceContext.from_defaults(
    chunk_size=1000,
)
```

More details at [customizing service context](/how_to/customization/service_context.md)

**Configuring LLM**
```python
from llama_index import ServiceContext
from llama_index.llms import PaLM

service_context = ServiceContext.from_defaults(
    llm=PaLM()
)
```

More details at [customizing LLMs](/how_to/customization/custom_llms.md)

## Data/Index Modules
**Set document metadata**
```python
document = Document(
    text='text', 
    metadata={
        'filename': '<doc_file_name>', 
        'category': '<category>'
    }
)
```
More details at [customizing documents](/how_to/customization/custom_documents.md)

**Configuring vector store**
```python
import chromadb
from llama_index import VectorStoreIndex
from llama_index.vector_stores import ChromaVectorStore

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("quickstart")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
```

More details at [customizing storage](/how_to/storage/customization.md).

## Query Modules
**Configuring number of text chunks to retrieve as context**
```python
query_engine = index.as_query_engine(
    similarity_top_k=1,
)
```

**Configuring response mode**
```python
query_engine = index.as_query_engine(
    response_mode='tree_summarize',
)
```
 
More details at query engine [usage pattern](/how_to/query_engine/usage_pattern.md) and [response modes](/how_to/query_engine/response_modes.md)

**Enabling streaming**
```python
query_engine = index.as_query_engine(
    streaming=True,
)
```

More details at [streaming guide](/how_to/customization/streaming.md)