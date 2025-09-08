# llama-index-retrievers-alletra-x10000

### This llama-index retriever will provide interfaces for the user applications to do semantic similarity search on the embeddings for Data Intelligence solution of HPE AlletraMP X10000

## How to use SDK

### Install llama-index-retriver-alletra-x10000

```python
pip install llama-index-retriever-alletra-x10000
```

### Data Intelligence solution supports inline embedding generation
``` python
# Add steps to enable inline vector embedding generation
# Create pipeline
# Create collection
# Add bucket to collection
```


### Driver code
- Feel free to change embedding and llm models.

```python
from llama_index.retrievers.data_intelligence import DataIntelligenceRetriever
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core import Settings
# TODO import python data intelligence client library


# Settings.llm = None
Settings.llm = HuggingFaceLLM(
    model_name="HuggingFaceH4/zephyr-7b-beta",
    tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
    context_window=3900,
    max_new_tokens=25,
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="auto",
)

retriever = DataIntelligenceRetriever(uri="https://example.com",
                                      s3_access_key="",
                                      s3_secret_key="",
                                      collection_name="testcollection", 
                                      top_k=50,
                                      search_config={
                                            "radius": 0.75
                                        }
                                      )
memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
chatbot = CondensePlusContextChatEngine.from_defaults(retriever=retriever, memory=memory)

res = chatbot.chat("Who was the eldest daughter of the steward of the old Lord de Versely?")
print(res)
```