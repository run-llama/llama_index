定义LLM
LlamaIndex的目标是提供一组数据结构，以便以LLM的提示限制易于兼容的方式组织外部信息。因此，LLM总是用于构建最终答案。根据所使用的[索引类型](/reference/indices.rst)，LLM也可能在索引构建、插入和查询遍历期间使用。

LlamaIndex使用Langchain的[LLM](https://python.langchain.com/en/latest/modules/models/llms.html)和[LLMChain](https://langchain.readthedocs.io/en/latest/modules/chains.html)模块来定义底层抽象。我们引入了一个包装类[`LLMPredictor`](/reference/service_context/llm_predictor.rst)，用于集成到LlamaIndex中。

我们还引入了一个[`PromptHelper`类](/reference/service_context/prompt_helper.rst)，以允许用户明确设置某些约束参数，例如最大输入大小（davinci模型的默认值为4096）、生成的输出令牌数、最大块重叠等等。

默认情况下，我们使用OpenAI的`text-davinci-003`模型。但是您可以选择自定义所使用的底层LLM。

下面我们展示一些LLM自定义的示例。这包括

- 更改底层LLM
- 更改输出令牌数（用于OpenAI、Cohere或AI21）
- 对任何LLM的所有参数（从输入大小到块重叠）进行更细粒度的控制

## 示例：更改底层LLM

下面展示了自定义所使用的LLM的示例代码片段。
在这个例子中，我们使用`text-davinci-002`而不是`text-davinci-003`。可用的模型包括`text-davinci-003`、`text-curie-001`、`text-babbage-001`、`text-ada-001`、`code-davinci-002`和`code-cushman-001`。请注意，您可以插入Langchain的[LLM](https://langchain.readthedocs.io/en/latest/modules/llms.html)页面上显示的任何LLM。

```python

from llama_index import (
    GPTKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext
)
from langchain import OpenAI

documents = SimpleDirectoryReader('data').load_data()

# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# build index
index = GPTKeywordTableIndex.from_documents(documents, service_context=service_context)

# get response from query
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do after his time at Y Combinator?")

```

## 示例：更改输出令牌数（用于OpenAI，Cohere，AI21）

默认情况下，输出令牌的数量通常被设置为一些较低的数字（例如，OpenAI的默认值为256）。

对于OpenAI，Cohere，AI21，您只需要设置`max_tokens`参数（或AI21的maxTokens）。我们将在后台处理文本分块/计算。StableLM由StabilityAI开发。
StableLM很高兴能够帮助用户，但是拒绝做任何可能对用户有害的事情。
StableLM不仅仅是一个信息源，StableLM还能够写诗、短篇小说，并开玩笑。
StableLM拒绝参与任何可能伤害人类的事情。使用此方法，您可以使用任何LLM。也许您有一个在本地运行，或者在您自己的服务器上运行。只要实现了类并返回了生成的令牌，它就应该起作用。请注意，我们需要使用提示助手来自定义提示大小，因为每个模型的上下文长度都略有不同。

请注意，您可能需要调整内部提示以获得良好的性能。即使如此，您也应该使用足够大的LLM来确保它能够处理LlamaIndex内部使用的复杂查询，因此您的里程可能会有所不同。

可以在[这里](https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/default_prompts.py)查看所有默认内部提示，聊天特定的提示列在[这里](https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/chat_prompts.py)，您还可以按照[这里](https://gpt-index.readthedocs.io/en/latest/how_to/customization/custom_prompts.html)描述的方式实现自定义提示。