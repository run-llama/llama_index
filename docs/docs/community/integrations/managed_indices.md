# Using Managed Indices

LlamaIndex offers multiple integration points with Managed Indices. A managed index is a special type of index that is not managed locally as part of LlamaIndex but instead is managed via an API, such as [Vectara](https://vectara.com).

## Using a Managed Index

Similar to any other index within LlamaIndex (tree, keyword table, list), any `ManagedIndex` can be constructed with a collection
of documents. Once constructed, the index can be used for querying.

If the Index has been previously populated with documents - it can also be used directly for querying.

## Google Generative Language Semantic Retriever

Google's Semantic Retrieve provides both querying and retrieval capabilities. Create a managed index, insert documents, and use a query engine or retriever anywhere in LlamaIndex!

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.indices.managed.google import GoogleIndex

# Create a corpus
index = GoogleIndex.create_corpus(display_name="My first corpus!")
print(f"Newly created corpus ID is {index.corpus_id}.")

# Ingestion
documents = SimpleDirectoryReader("data").load_data()
index.insert_documents(documents)

# Querying
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

# Retrieving
retriever = index.as_retriever()
source_nodes = retriever.retrieve("What did the author do growing up?")
```

See the [notebook guide](../../examples/managed/GoogleDemo.ipynb) for full details.

## Vectara

First, [sign up](https://vectara.com/integrations/llama_index) and use the Vectara Console to create a corpus (aka Index), and add an API key for access.
Once you have your API key, export it as an environment variable:

```python
import os

os.environ["VECTARA_API_KEY"] = "<YOUR_VECTARA_API_KEY>"
os.environ["VECTARA_CORPUS_KEY"] = "<YOUR_VECTARA_CORPUS_KEY>"
```

Then construct the Vectara Index and query it as follows:

```python
from llama_index.core import ManagedIndex, SimpleDirectoryReade
from llama_index.indices.managed.vectara import VectaraIndex

# Load documents and build index
vectara_corpus_key = os.environ.get("VECTARA_CORPUS_KEY")
vectara_api_key = os.environ.get("VECTARA_API_KEY")

documents = SimpleDirectoryReader("../paul_graham_essay/data").load_data()
index = VectaraIndex.from_documents(
    documents,
    vectara_corpus_key=vectara_corpus_key,
    vectara_api_key=vectara_api_key,
)
```

Notes:
* If the environment variables `VECTARA_CORPUS_KEY` and `VECTARA_API_KEY` are in the environment already, you do not have to explicitly specify them in your call and the VectaraIndex class will read them from the environment.
* To connect to multiple Vectara corpora, you can set `VECTARA_CORPUS_KEY` to a comma-separated list, for example: `12,51` would connect to corpus `12` and corpus `51`.

If you already have documents in your corpus, you can just access the data directly by constructing the `VectaraIndex` as follows:

```python
index = VectaraIndex()
```

The VectaraIndex will connect to the existing corpus without loading any new documents.

To query the index, simply construct a query engine as follows:

```python
query_engine = index.as_query_engine(summary_enabled=True)
print(query_engine.query("What did the author do growing up?"))
```

Or you can use the chat functionality:

```python
chat_engine = index.as_chat_engine()
print(chat_engine.chat("What did the author do growing up?").response)
```

Chat works as you expect where subsequent `chat` calls maintain a conversation history. All of this is done on the Vectara platform so you don't have to add any additional logic.

For more examples - please see below:

- [Vectara Demo](../../examples/managed/vectaraDemo.ipynb)
- [Vectara AutoRetriever](../../examples/retrievers/vectara_auto_retriever.ipynb)

## Vertex AI RAG (LlamaIndex on Vertex AI)

[LlamaIndex on Vertex AI for RAG](https://cloud.google.com/vertex-ai/generative-ai/docs/llamaindex-on-vertexai) is a managed RAG index on Google Cloud Vertex AI.

First, [create a Google Cloud project and enable the Vertex AI API](https://cloud.google.com/vertex-ai/docs/start/cloud-environment). Then run the following code to create a managed index.

```python
from llama_index.indices.managed.vertexai import VertexAIIndex

# TODO(developer): Replace these values with your project information
project_id = "YOUR_PROJECT_ID"
location = "us-central1"

# Optional: If using an existing corpus
corpus_id = "YOUR_CORPUS_ID"

# Optional: If creating a new corpus
corpus_display_name = "my-corpus"
corpus_description = "Vertex AI Corpus for LlamaIndex"

# Create a corpus or provide an existing corpus ID
index = VertexAIIndex(
    project_id,
    location,
    corpus_display_name=corpus_display_name,
    corpus_description=corpus_description,
)
print(f"Newly created corpus name is {index.corpus_name}.")

# Import files from Google Cloud Storage or Google Drive
index.import_files(
    uris=["https://drive.google.com/file/123", "gs://my_bucket/my_files_dir"],
    chunk_size=512,  # Optional
    chunk_overlap=100,  # Optional
)

# Upload local file
index.insert_file(
    file_path="my_file.txt",
    metadata={"display_name": "my_file.txt", "description": "My file"},
)

# Querying
query_engine = index.as_query_engine()
response = query_engine.query("What is RAG and why it is helpful?")

# Retrieving
retriever = index.as_retriever()
nodes = retriever.retrieve("What is RAG and why it is helpful?")
```

See the [notebook guide](../../examples/managed/VertexAIDemo.ipynb) for full details.
