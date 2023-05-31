# Chat Engines

Chat engine is a high-level interface for having a conversation with your data
(multiple back-and-forth instead of a single question & answer).  
Think ChatGPT, but augmented with your knowledge base.  

Conceptually, it is a stateful analogous of a query engine.  
By keeping track of the conversation history, it can answer questions with past context in mind.  

We provide a few simple implementations to start, with more sophisticated modes coming soon!
More specifically, the `SimpleChatEngine` does not make use of a knowledge base, 
whereas `CondenseQuestionChatEngine` and `ReActChatEngine` make use of a query engine over knowledge base.

Configuring a chat engine is very similar to configuring a query engine.

In the high-level API:
```python
chat_engine = index.as_chat_engine(chat_mode='condense_question', verbose=True)
```
> Note: you can access different chat engines by specifying the `chat_mode` as a kwarg. `condense_question` corresponds to `CondenseQuestionChatEngine`, `react` corresponds to `ReActChatEngine`.

In the low-level composition API:
```python
query_engine = index.as_query_engine()
chat_engine = CondenseQuestionChatEngine.from_defaults(query_engine, verbose=True)
```

Below you can find corresponding tutorials to see the available chat engines in action. 

```{toctree}
---
caption: Examples
maxdepth: 1
---
../../examples/chat_engine/chat_engine_repl.ipynb
../../examples/chat_engine/chat_engine_condense_question.ipynb
../../examples/chat_engine/chat_engine_react.ipynb
```