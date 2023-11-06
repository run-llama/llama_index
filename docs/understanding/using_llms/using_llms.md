# Using LLMs

```{tip}
For a list of our supported LLMs and a comparison of their functionality, check out our [LLM module guide](/module_guides/models/llms.md).
```

One of the first steps when building an LLM-based application is which LLM to use; you can also use more than one if you wish.

LLMs are used at multiple different stages of your pipeline:

- During **Indexing** you may use an LLM to determine the relevance of data (whether to index it at all) or you may use an LLM to summarize the raw data and index the summaries instead.
- During **Querying** LLMs can be used in two ways:
  - During **Retrieval** (fetching data from your index) LLMs can be given an array of options (such as multiple different indices) and make decisions about where best to find the information you're looking for. An agentic LLM can also use _tools_ at this stage to query different data sources.
  - During **Response Synthesis** (turning the retrieved data into an answer) an LLM can combine answers to multiple sub-queries into a single coherent answer, or it can transform data, such as from unstructured text to JSON or another programmatic output format.

LlamaIndex provides a single interface to a large number of different LLMs, allowing you to pass in any LLM you choose to any stage of the pipeline. It can be as simple as this:

```python
from llama_index.llms import OpenAI

response = OpenAI().complete("Paul Graham is ")
print(response)
```

Usually you will instantiate an LLM and pass it to a `ServiceContext`, which you then pass to other stages of the pipeline, as in this example:

```python
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

llm = OpenAI(temperature=0.1, model="gpt-4")
service_context = ServiceContext.from_defaults(llm=llm)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)
```

In this case, you've instantiated OpenAI and customized it to use the `gpt-4` model instead of the default `gpt-3.5-turbo`, and also modified the `temperature`. The `VectorStoreIndex` will now use gpt-4 to encode or `embed` your documents for later querying.

```{tip}
A ServiceContext is a bundle of configuration data that you pass into different parts of LlamaIndex. You can [learn more about ServiceContext](/module_guides/supporting_modules/service_context.md) and how to customize it, including using multiple ServiceContexts to use multiple LLMs.
```

## Available LLMs

We support integrations with OpenAI, Hugging Face, PaLM, and more. Check out our [module guide to LLMs](/module_guides/models/llms.md) for a full list, including how to run a local model.

### Using a local LLM

LlamaIndex doesn't just supported hosted LLM APIs; you can also [run a local model such as Llama2 locally](https://replicate.com/blog/run-llama-locally).

Once you have a local LLM such as Llama 2 installed, you can use it like this:

```python
from llama_index import ServiceContext

service_context = ServiceContext.from_defaults(llm="local")
```

This will use llama2-chat-13B from with LlamaCPP, and assumes you have `llama-cpp-python` installed. A [full LlamaCPP usage guide is available](/examples/llm/llama_2_llama_cpp.ipynb).

See the [custom LLM's How-To](/module_guides/models/llms/usage_custom.md) for more details.

## Prompts

By default LlamaIndex comes with a great set of built-in, battle-tested prompts that handle the tricky work of getting a specific LLM to correctly handle and format data. This is one of the biggest benefits of using LlamaIndex. If you want to, you can [customize the prompts](/module_guides/models/prompts.md)

```{toctree}
---
maxdepth: 1
hidden: true
---
/understanding/using_llms/privacy.md
```
