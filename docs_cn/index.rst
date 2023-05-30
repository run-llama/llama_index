欢迎来到LlamaIndex🦙！
=====================================

LlamaIndex（GPT Index）是一个用于您的LLM应用程序的数据框架。

- Github：https://github.com/jerryjliu/llama_index
- PyPi：
   - LlamaIndex：https://pypi.org/project/llama-index/。
   - GPT Index（重复）：https://pypi.org/project/gpt-index/。
- Twitter：https://twitter.com/llama_index
- Discord https://discord.gg/dGcwcsnxhU

生态系统
^^^^^^^^^

- 🏡LlamaHub：https://llamahub.ai
- 🧪LlamaLab：https://github.com/run-llama/llama-lab


🚀概览
-----------

上下文
^^^^^^^
- LLMs是一种用于知识生成和推理的非凡技术。它们已经在大量公开可用的数据上进行了预训练。
- 我们如何最好地用自己的私有数据来增强LLMs？

我们需要一个全面的工具包来帮助执行这种数据增强LLMs。


提出的解决方案
^^^^^^^^^^^^^^^^^
这就是**LlamaIndex**的用武之地。 LlamaIndex是一个“数据框架”，可帮助您构建LLM应用程序。它提供以下工具：

- 提供**数据连接器**以摄取您现有的数据源和数据格式（API，PDF，文档，SQL等）
- 提供结构化您的数据（索引，图）的方法，以便可以轻松地与LLMs一起使用此数据。
- 提供**高级检索/查询界面**：输入任何LLM输入提示，返回检索的上下文和知识增强的输出。
- 允许与外部应用程序框架（例如LangChain，Flask，Docker，ChatGPT等）轻松集成。

LlamaIndex为初存用户和高级用户提供工具。我们的高级API允许初存用户使用LlamaIndex在5行代码中摄取和查询其数据。我们的低级API允许高级用户自定义和扩展任何模块（数据连接器，索引，检索器，查询引擎，重新排序模块），以满足其需求。.. toctree::
   :maxdepth: 1
   :caption: 参考
   :hidden:

   reference/indices.rst
   reference/query.rst
   reference/node.rst
   reference/llm_predictor.rst
   reference/node_postprocessor.rst
   reference/storage.rst
   reference/composability.rst
   reference/readers.rst
   reference/prompts.rst
   reference/service_context.rst
   reference/optimizers.rst
   reference/callbacks.rst
   reference/struct_store.rst
   reference/response.rst
   reference/playground.rst
   reference/node_parser.rst
   reference/example_notebooks.rst
   reference/langchain_integrations/base.rst


.. toctree::
   :maxdepth: 1
   :caption: 画廊
   :hidden:

   gallery/app_showcase.md