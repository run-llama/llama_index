聊天引擎是一种高级界面，可以与您的数据进行对话（多次来回而不是单个问题和答案）。想象一下ChatGPT，但增强了您的知识库。从概念上讲，它是查询引擎的有状态类比。通过跟踪对话历史，它可以考虑过去的背景来回答问题。

我们提供了一些简单的实现来开始，很快就会有更多复杂的模式！更具体地说，`SimpleChatEngine`不使用知识库，而`CondenseQuestionChatEngine`和`ReActChatEngine`则使用查询引擎来查询知识库。

配置聊天引擎与配置查询引擎非常相似。

在高级API中：
```python
chat_engine = index.as_chat_engine(chat_mode='condense_question', verbose=True)
```
>注意：您可以通过指定`chat_mode`作为kwarg来访问不同的聊天引擎。`condense_question`对应于`CondenseQuestionChatEngine`，`react`对应于`ReActChatEngine`。

在低级组合API中：
```python
query_engine = index.as_query_engine()
chat_engine = CondenseQuestionChatEngine.from_defaults(query_engine, verbose=True)
```

在下面，您可以找到相应的教程，以查看可用的聊天引擎。

```{toctree}
---
caption: Examples
maxdepth: 1
---
../../examples/chat_engine/chat_engine_repl.ipynb
../../examples/chat_engine/chat_engine_condense_question.ipynb
../../examples/chat_engine/chat_engine_react.ipynb
```