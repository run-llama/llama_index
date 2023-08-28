# Usage Pattern

## Getting Started

The most common usage for an embedding model will be setting it in the service context object, and then using it to construct an index and query. The input documents will be broken into nodes, and the emedding model will generate an embedding for each node.

By default, LlamaIndex will use `text-embedding-ada-002`, which is what the example below manually sets up for you.

```python
from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings import OpenAIEmbedding

embed_model = OpenAIEmbedding()
service_context = serviceContext.from_defaults(embed_model=embed_model)

# optionally set a global service context to avoid passing it into other objects every time
from llama_index import set_global_service_context
set_global_service_context(service_context)

documents = SimpleDirectoryReader("./data").load_data()

index = VectorStoreIndex.from_documents(documents)
```

Then, at query time, the embedding model will be used again to embed the query text.

```python
query_engine = index.as_query_engine()

response = query_engine.query("query string")
```

## Customization

### Batch Size

By default, embeddings requests are sent to OpenAI in batches of 10. For some users, this may (rarely) incur a rate limit. For other users embedding many documents, this batch size may be too small.

```python
# set the batch size to 42
embed_model = OpenAIEmbedding(embed_batch_size=42)
```

(local-embedding-models)=

### Local Embedding Models

The easiest way to use a local model is:

```python
from llama_index import ServiceContext
service_context = ServiceContext.from_defaults(embed_model="local")
```

To configure the model used (from Hugging Face hub), add the model name separated by a colon:

```python
from llama_index import ServiceContext

service_context = ServiceContext.from_defaults(
  embed_model="local:BAAI/bge-large-en"
)
```

### Embedding Model Integrations

We also support any embeddings offered by Langchain [here](https://python.langchain.com/docs/modules/data_connection/text_embedding/).

The example below loads a model from Hugging Face, using Langchain's embedding class.

```python
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index import ServiceContext

embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")

service_context = ServiceContext.from_defaults(embed_model=embed_model)
```

### Custom Embedding Model

If you wanted to use embeddings not offered by LlamaIndex or Langchain, you can also extend our base embeddings class and implement your own!

The example below uses Instructor Embeddings ([install/setup details here](https://huggingface.co/hkunlp/instructor-large)), and implements a custom embeddings class. Instructor embeddings work by providing text, as well as "instructions" on the domain of the text to embed. This is helpful when embedding text from a very specific and specialized topic.

```python
from typing import Any, List
from InstructorEmbedding import INSTRUCTOR
from llama_index.embeddings.base import BaseEmbedding

class InstructorEmbeddings(BaseEmbedding):
  def __init__(
    self, 
    instructor_model_name: str = "hkunlp/instructor-large",
    instruction: str = "Represent the Computer Science documentation or question:",
    **kwargs: Any,
  ) -> None:
    self._model = INSTRUCTOR(instructor_model_name)
    self._instruction = instruction
    super().__init__(**kwargs)

    def _get_query_embedding(self, query: str) -> List[float]:
      embeddings = self._model.encode([[self._instruction, query]])
      return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
      embeddings = self._model.encode([[self._instruction, text]])
      return embeddings[0] 

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
      embeddings = self._model.encode([[self._instruction, text] for text in texts])
      return embeddings
```

## Standalone Usage

You can also use embeddings as a standalone module for your project, existing application, or general testing and exploration.

```python
embeddings = embed_model.get_text_embedding("It is raining cats and dogs here!")
```
