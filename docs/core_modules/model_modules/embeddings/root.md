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

```{toctree}
---
maxdepth: 1
---
usage_pattern.md
```

## Modules

We support integrations with OpenAI, Azure, and anything LangChain offers. Details below.

```{toctree}
---
maxdepth: 1
---
modules.md
```
