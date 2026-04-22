# LlamaIndex Retrievers Integration: AlletraX10000Retriever

This llama-index retriever will provide interfaces for the user applications to do semantic similarity search on the embeddings for Data Intelligence solution of HPE AlletraMP X10000

## How to use SDK

### Install llama-index-retrievers-alletra-x10000-retriever

```sh
pip install llama-index-retrievers-alletra-x10000-retriever
```

### Driver code

```python
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.retrievers.alletra_x10000_retriever import (
    AlletraX10000Retriever,
)


# Configure the llm settings before using the retriever

retriever = AlletraX10000Retriever(
    uri="https://example.com",
    s3_access_key="",
    s3_secret_key="",
    collection_name="testcollection",
    top_k=50,
    search_config={"radius": 0.75},
)

memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
chatbot = CondensePlusContextChatEngine.from_defaults(
    retriever=retriever, memory=memory
)

res = chatbot.chat(
    "Who was the eldest daughter of the steward of the old Lord de Versely?"
)
print(res)
```
