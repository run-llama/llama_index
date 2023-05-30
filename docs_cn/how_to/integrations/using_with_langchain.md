使用Langchain 🦜🔗

LlamaIndex提供了用于Langchain代理的工具抽象以及内存模块。

工具抽象+内存模块的API参考[在这里](/reference/langchain_integrations/base.rst)。

### Llama Tool abstractions
LlamaIndex提供了工具抽象，因此您可以使用LlamaIndex与Langchain代理一起使用。

例如，您可以选择直接从`QueryEngine`创建“Tool”，如下所示：

```python
from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool

tool_config = IndexToolConfig(
    query_engine=query_engine, 
    name=f"Vector Index",
    description=f"useful for when you want to answer queries about X",
    tool_kwargs={"return_direct": True}
)

tool = LlamaIndexTool.from_tool_config(tool_config)

```

您还可以选择提供一个`LlamaToolkit`：

```python
toolkit = LlamaToolkit(
    index_configs=index_configs,
)
```

这样的工具包可以用于通过我们的`create_llama_agent`和`create_llama_chat_agent`命令创建下游基于Langchain的聊天代理：

```python
from llama_index.langchain_helpers.agents import create_llama_chat_agent

agent_chain = create_llama_chat_agent(
    toolkit,
    llm,
    memory=memory,
    verbose=True
)

agent_chain.run(input="Query about X")
```

您可以在[这里查看完整的教程笔记本](https://github.com/jerryjliu/llama_index/blob/main/examples/chatbot/Chatbot_SEC.ipynb)。

### Llama Demo Notebook：Tool + Memory module

我们还提供了另一个演示笔记本，演示如何使用以下组件构建聊天代理。
- 使用LlamaIndex作为具有Langchain代理的通用可调用工具
- 使用LlamaIndex作为内存模块；这允许您使用Langchain聊天机器人插入任意数量的对话历史！

请参阅[此处的笔记本](https://github.com/jerryjliu/llama_index/blob/main/examples/langchain_demo/LangchainDemo.ipynb)。