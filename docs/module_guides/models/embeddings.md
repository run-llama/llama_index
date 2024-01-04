# Embeddings

## Concept

Embeddings are used in LlamaIndex to represent your documents using a sophisticated numerical representation. Embedding models take text as input, and return a long list of numbers used to capture the semantics of the text. These embedding models have been trained to represent text this way, and help enable many applications, including search!

At a high level, if a user asks a question about dogs, then the embedding for that question will be highly similar to text that talks about dogs.

When calculating the similarity between embeddings, there are many methods to use (dot product, cosine similarity, etc.). By default, LlamaIndex uses cosine similarity when comparing embeddings.

There are many embedding models to pick from. By default, LlamaIndex uses `text-embedding-ada-002` from OpenAI. We also support any embedding model offered by Langchain [here](https://python.langchain.com/docs/modules/data_connection/text_embedding/), as well as providing an easy to extend base class for implementing your own embeddings.

## Usage Pattern

Most commonly in LlamaIndex, embedding models will be specified in the `ServiceContext` object, and then used in a vector index. The embedding model will be used to embed the documents used during index construction, as well as embedding any queries you make using the query engine later on.

```python
from llama_index import ServiceContext
from llama_index.embeddings import OpenAIEmbedding

embed_model = OpenAIEmbedding()
service_context = ServiceContext.from_defaults(embed_model=embed_model)
```

To save costs, you may want to use a local model.

```python
from llama_index import ServiceContext

service_context = ServiceContext.from_defaults(embed_model="local")
```

This will use a well-performing and fast default from Hugging Face.

You can find more usage details and available customization options below.

## Getting Started

The most common usage for an embedding model will be setting it in the service context object, and then using it to construct an index and query. The input documents will be broken into nodes, and the embedding model will generate an embedding for each node.

By default, LlamaIndex will use `text-embedding-ada-002`, which is what the example below manually sets up for you.

```python
from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings import OpenAIEmbedding

embed_model = OpenAIEmbedding()
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# Optionally set a global service context to avoid passing it into other objects every time
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

### HuggingFace Optimum ONNX Embeddings

LlamaIndex also supports creating and using ONNX embeddings using the Optimum library from HuggingFace. Simple create and save the ONNX embeddings, and use them.

Some prerequisites:

```
pip install transformers optimum[exporters]
```

Creation with specifying the model and output path:

```python
from llama_index.embeddings import OptimumEmbedding

OptimumEmbedding.create_and_save_optimum_model(
    "BAAI/bge-small-en-v1.5", "./bge_onnx"
)
```

And then usage:

```python
embed_model = OptimumEmbedding(folder_name="./bge_onnx")
service_context = ServiceContext.from_defaults(embed_model=embed_model)
```

### LangChain Integrations

We also support any embeddings offered by Langchain [here](https://python.langchain.com/docs/modules/data_connection/text_embedding/).

The example below loads a model from Hugging Face, using Langchain's embedding class.

```python
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index import ServiceContext

embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")

service_context = ServiceContext.from_defaults(embed_model=embed_model)
```

(custom_embeddings)=

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
            embeddings = self._model.encode(
                [[self._instruction, text] for text in texts]
            )
            return embeddings
```

## Standalone Usage

You can also use embeddings as a standalone module for your project, existing application, or general testing and exploration.

```python
embeddings = embed_model.get_text_embedding(
    "It is raining cats and dogs here!"
)
```

(list_of_embeddings)=

## List of supported embeddings

We support integrations with OpenAI, Azure, and anything LangChain offers.

```{toctree}
---
maxdepth: 1
---
/examples/embeddings/OpenAI.ipynb
/examples/embeddings/Langchain.ipynb
/examples/embeddings/cohereai.ipynb
/examples/embeddings/fastembed.ipynb
/examples/embeddings/gradient.ipynb
/examples/customization/llms/AzureOpenAI.ipynb
/examples/embeddings/custom_embeddings.ipynb
/examples/embeddings/huggingface.ipynb
/examples/embeddings/elasticsearch.ipynb
/examples/embeddings/clarifai.ipynb
/examples/embeddings/llm_rails.ipynb
/examples/embeddings/text_embedding_inference.ipynb
/examples/embeddings/google_palm.ipynb
/examples/embeddings/jinaai_embeddings.ipynb
/examples/embeddings/voyageai.ipynb
/examples/embeddings/mistralai.ipynb
```
