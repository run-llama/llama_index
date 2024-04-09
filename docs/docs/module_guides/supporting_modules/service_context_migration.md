# Migrating from ServiceContext to Settings

Introduced in v0.10.0, there is a new global `Settings` object intended to replace the old `ServiceContext` configuration.

The new `Settings` object is a global settings, with parameters that are lazily instantiated. Attributes like the LLM or embedding model are only loaded when they are actually required by an underlying module.

Previously with the service context, various modules often did not use it, and it also forced loading every component into memory at runtime (even if those components weren't used).

Configuring the global settings means you are change the default for EVERY module in LlamaIndex. This means if you aren't using OpenAI, an example config might look like:

```python
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.llm = Ollama(model="llama2", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
```

Now with this settings, you can ensure OpenAI will never be used in the framework.

The `Settings` object supports nearly all the same attributes as the old `ServiceConext`. A complete list can be found in the [docs page](settings.md).

### Complete Migration

Below is an example of completely migrating from `ServiceContext` to `Settings`:

**Before**

```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core import ServiceContext, set_global_service_context

service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo"),
    embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
    node_parser=SentenceSplitter(chunk_size=512, chunk_overlap=20),
    num_output=512,
    context_window=3900,
)
set_global_service_context(service_context)
```

**After**

```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900
```

## Local Config

The above covers global configuration. To config settings per-module, all module interfaces should be updated to accept kwargs for the objects that are being used.

If you are using an IDE, the kwargs should auto-populate with intellisense, but here are some examples below:

```python
# a vector store index only needs an embed model
index = VectorStoreIndex.from_documents(
    documents, embed_model=embed_model, transformations=transformations
)

# ... until you create a query engine
query_engine = index.as_query_engine(llm=llm)
```

```python
# a document summary index needs both an llm and embed model
# for the constructor
index = DocumentSummaryIndex.from_documents(
    documents, embed_model=embed_model, llm=llm
)
```
