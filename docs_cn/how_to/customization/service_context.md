ServiceContext对象封装了用于创建索引和运行查询的资源。

可以在服务上下文中设置以下可选项：

- llm_predictor：用于生成对查询的自然语言响应的LLM。
- embed_model：用于生成文本的向量表示的嵌入模型。
- prompt_helper：定义发送到LLM的文本设置的PromptHelper对象。
- node_parser：将文档转换为节点的解析器。
- chunk_size_limit：节点的最大大小。用于提示助手和节点解析器，当它们没有提供时。
- callback_managaer：在事件上调用其处理程序的回调管理器对象。提供基本的日志记录和跟踪功能。

以下是使用默认设置设置所有对象的完整示例：

```python
from langchain.llms import OpenAI
from llama_index import ServiceContext, LLMPredictor, OpenAIEmbedding, PromptHelper
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser

llm_predictor = LLMPredictor(llm=OpenAI(model_name='text-davinci-003', temperature=0))
embed_model = OpenAIEmbedding()
node_parser = SimpleNodeParser(text_splitter=TokenTextSplitter(chunk_size=1024, chunk_overlap=200))
prompt_helper = PromptHelper(max_input_size=4096, num_output=256, max_chunk_overlap=20, chunk_size_limit=1024)
service_context = ServiceContext.from_defaults(
  llm_predictor=llm_predictor,
  embed_model=embed_model,
  node_parser=node_parser,
  prompt_helper=prompt_helper
)
```

## 全局ServiceContext

您可以通过设置全局服务上下文来为ServiceContext指定不同的默认值。

使用全局服务上下文时，调用`ServiceContext.from_defaults()`时未提供的任何属性都将从您的全局服务上下文中提取。如果您从未在其他地方定义服务上下文，则始终使用全局服务上下文。

以下是全局服务上下文的快速示例。此服务上下文将LLM更改为`gpt-3.5-turbo`，更改`chunk_size_limit`，并使用`LlamaDebugHandler`跟踪事件设置`callback_manager`。

首先，定义服务上下文：

```python
from langchain.chat_models import ChatOpenAI
from llama_index import ServiceContext, LLMPredictor
from llama_index.callbacks import CallbackManager, LlamaDebugHandler

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7)
llm_predictor = LLMPredictor(llm=llm)
service_context = ServiceContext(
  llm_predictor=llm_predictor,
  chunk_size_limit=2048,
  callback_manager=callback_manager
)
```

ServiceContext封装了用于创建索引和运行查询的资源。可以在服务上下文中设置以下可选项：llm_predictor：用于生成对查询的自然语言响应的LLM；embed_model：用于生成文本的向量表示的嵌入模型；prompt_helper：定义发送到LLM的文本设置的PromptHelper对象；node_parser：将文档转换为节点的解析器；chunk_size_limit：节点的最大大小；callback_managaer：在事件上调用其处理程序的回调管理器对象，提供基本的日志记录和跟踪功能。可以通过设置全局服务上下文来为ServiceContext指定不同的默认值。使用全局服务上下文时，调用`ServiceContext.from_defaults()`时未提供的任何属性都将从您的全局服务上下文中提取。然后，设置全局服务上下文对象：
```python
from llama_index import set_global_service_context
set_global_service_context（service_context）
```