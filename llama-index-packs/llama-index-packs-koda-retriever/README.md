# Koda Retriever

This retriever is a custom fine-tunable Hybrid Retriever that dynamically determines the optimal alpha for a given query.
An LLM is used to categorize the query and therefore determine the optimal alpha value, as each category has a preset/provided alpha value.
It is recommended that you run tests on your corpus of data and queries to determine categories and corresponding alpha values for your use case.

![koda-retriever-mascot](https://i.imgur.com/224ocIw.jpeg)

### Disclaimer

_The default categories and alpha values are not recommended for production use_

## Introduction

Alpha tuning in hybrid retrieval for RAG models refers to the process of adjusting the weight (alpha) given to different components of a hybrid search strategy. In RAG, the retrieval component is crucial for fetching relevant context from a knowledge base, which the generation component then uses to produce answers. By fine-tuning the alpha parameter, the balance between the retrieved results from dense vector search methods and traditional sparse methods can be optimized. This optimization aims to enhance the overall performance of the system, ensuring that the retrieval process effectively supports the generation of accurate and contextually relevant responses.

### Simply explained

Imagine you're playing a game where someone whispers a sentence to you, and you have to decide whether to draw a picture of exactly what they said, or draw a picture of what you think they mean. Alpha tuning is like finding the best rule for when to draw exactly what's said and when to think deeper about the meaning. It helps us get the best mix, so the game is more fun and everyone understands each other better!

## Usage Snapshot

Koda Retriever is compatible with all other retrieval interfaces and objects that would normally be able to interact with an LI-native [retriever](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/root.html).

Please see the [examples](./examples/) folder for more specific examples.

```python
# Setup
from llama_index.packs.koda_retriever import KodaRetriever
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.postprocessor import LLMRerank
from llama_index.core import Settings

Settings.llm = OpenAI()
Settings.embed_model = OpenAIEmbedding()
vector_store = PineconeVectorStore(pinecone_index=index, text_key="summary")
vector_index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, embed_model=Settings.embed_model
)

reranker = LLMRerank(llm=Settings.llm)  # optional
retriever = KodaRetriever(
    index=vector_index, llm=Settings.llm, reranker=reranker, verbose=True
)

# Retrieval
query = "What was the intended business model for the parks in the Jurassic Park lore?"

results = retriever.retrieve(query)

# Query Engine
query_engine = RetrieverQueryEngine.from_args(retriever=retriever)

response = query_engine.query(query)
```

### Prerequisites

- Vector Store Index w/ hybrid search enabled
- LLM (or any model to route/classify prompts)

Please note that you will also need vector AND text representations of your data for a hybrid retriever to work. It is not uncommon for some vector databases to only store the vectors themselves, in which case an error will occur downstream if you try to run any hybrid queries.

## Setup

## Citations

Idea & original implementation sourced from the following docs:

- https://blog.llamaindex.ai/llamaindex-enhancing-retrieval-performance-with-alpha-tuning-in-hybrid-search-in-rag-135d0c9b8a00
- https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/azure-ai-search-outperforming-vector-search-with-hybrid/ba-p/3929167

## Buy me a coffee

[Thanks!](https://www.buymeacoffee.com/nodice)
