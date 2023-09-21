# Chat Engine

## Concept
Chat engine is a high-level interface for having a conversation with your data
(multiple back-and-forth instead of a single question & answer).
Think ChatGPT, but augmented with your knowledge base.  

Conceptually, it is a **stateful** analogy of a [Query Engine](../query_engine/root.md). 
By keeping track of the conversation history, it can answer questions with past context in mind.  


```{tip}
If you want to ask standalone question over your data (i.e. without keeping track of conversation history), use [Query Engine](../query_engine/root.md) instead.  
```

## Usage Pattern
Get started with:
```python
chat_engine = index.as_chat_engine()
response = chat_engine.chat("Tell me a joke.")
```

To stream response:
```python
chat_engine = index.as_chat_engine()
streaming_response = chat_engine.stream_chat("Tell me a joke.")
for token in streaming_response.response_gen:
    print(token, end="")
```


```{toctree}
---
maxdepth: 2
---
usage_pattern.md
```


## Modules
Below you can find corresponding tutorials to see the available chat engines in action. 

```{toctree}
---
maxdepth: 2
---
modules.md
```