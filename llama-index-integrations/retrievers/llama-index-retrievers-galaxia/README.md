# LlamaIndex Retrievers Integration: Galaxia

## Galaxia Knowledge Base

> Galaxia Knowledge Base is an integrated knowledge base and retrieval mechanism for RAG. In contrast to standard solution, it is based on Knowledge Graphs built using symbolic NLP and Knowledge Representation solutions. Provided texts are analysed and transformed into Graphs containing text, language and semantic information. This rich structure allows for retrieval that is based on semantic information, not on vector similarity/distance.

Implementing RAG using Galaxia involves first uploading your files to [Galaxia](beta.cloud.smabbler.com/), analyzing them there and then building a model (knowledge graph). When the model is built, you can use `GalaxiaRetriever` to connect to the API and start retrieving.

## Installation

```
pip install llama-index-retrievers-galaxia
```

## Usage

```
from llama_index.retrievers.galaxia import GalaxiaRetriever
from llama_index.core.schema import QueryBundle

retriever = GalaxiaRetriever(
    api_url="https://beta.api.smabbler.com",
    api_key="<key>",
    knowledge_base_id="<knowledge_base_id>",
)

result = retriever._retrieve(QueryBundle(
    "<test question>"
))

print(result)

```
