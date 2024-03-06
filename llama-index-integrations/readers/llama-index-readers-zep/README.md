# Zep Reader

```bash
pip install llama-index-readers-zep
```

The Zep Reader returns a set of texts corresponding to a text query or embeddings retrieved from a Zep Collection.
The Reader is initialized with a Zep API URL and optionally an API key. The Reader can then be used to load data
from a Zep Document Collection.

## About Zep

Zep is a long-term memory store for LLM applications. Zep makes it simple to add relevant documents, chat history memory
and rich user data to your LLM app's prompts.

For more information about Zep and the Zep Quick Start Guide, see the [Zep documentation](https://docs.getzep.com/).

## Usage

Here's an end-to-end example usage of the ZepReader. First, we create a Zep Collection, chunk a document,
and add it to the collection.

We then wait for Zep's async embedder to embed the document chunks. Finally, we query the collection and print the
results.

```python
import time
from uuid import uuid4

from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import Document
from zep_python import ZepClient
from zep_python.document import Document as ZepDocument


from llama_index.readers.zep import ZepReader

# Create a Zep collection
zep_api_url = "http://localhost:8000"  # replace with your Zep API URL
collection_name = f"babbage{uuid4().hex}"
file = "babbages_calculating_engine.txt"

print(f"Creating collection {collection_name}")

client = ZepClient(base_url=zep_api_url, api_key="optional_api_key")
collection = client.document.add_collection(
    name=collection_name,  # required
    description="Babbage's Calculating Engine",  # optional
    metadata={"foo": "bar"},  # optional metadata
    embedding_dimensions=1536,  # this must match the model you've configured in Zep
    is_auto_embedded=True,  # use Zep's built-in embedder. Defaults to True
)

node_parser = SimpleNodeParser.from_defaults(chunk_size=250, chunk_overlap=20)

with open(file) as f:
    raw_text = f.read()

print("Splitting text into chunks and adding them to the Zep vector store.")
docs = node_parser.get_nodes_from_documents(
    [Document(text=raw_text)], show_progress=True
)

# Convert nodes to ZepDocument
zep_docs = [ZepDocument(content=d.get_content()) for d in docs]
uuids = collection.add_documents(zep_docs)
print(f"Added {len(uuids)} documents to collection {collection_name}")

print("Waiting for documents to be embedded")
while True:
    c = client.document.get_collection(collection_name)
    print(
        "Embedding status: "
        f"{c.document_embedded_count}/{c.document_count} documents embedded"
    )
    time.sleep(1)
    if c.status == "ready":
        break

query = "Was Babbage awarded a medal?"

# Using the ZepReader to load data from Zep
reader = ZepReader(api_url=zep_api_url, api_key="optional_api_key")
results = reader.load_data(
    collection_name=collection_name, query=query, top_k=3
)

print("\n\n".join([r.text for r in results]))
```
