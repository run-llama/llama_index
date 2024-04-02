# Using LLMs

!!! tip
    For a list of our supported LLMs and a comparison of their functionality, check out our [LLM module guide](../../module_guides/models/llms.md).

One of the first steps when building an LLM-based application is which LLM to use; you can also use more than one if you wish.

LLMs are used at multiple different stages of your pipeline:

- During **Indexing** you may use an LLM to determine the relevance of data (whether to index it at all) or you may use an LLM to summarize the raw data and index the summaries instead.
- During **Querying** LLMs can be used in two ways:
  - During **Retrieval** (fetching data from your index) LLMs can be given an array of options (such as multiple different indices) and make decisions about where best to find the information you're looking for. An agentic LLM can also use _tools_ at this stage to query different data sources.
  - During **Response Synthesis** (turning the retrieved data into an answer) an LLM can combine answers to multiple sub-queries into a single coherent answer, or it can transform data, such as from unstructured text to JSON or another programmatic output format.

LlamaIndex provides a single interface to a large number of different LLMs, allowing you to pass in any LLM you choose to any stage of the pipeline. It can be as simple as this:

```python
from llama_index.llms.openai import OpenAI

response = OpenAI().complete("Paul Graham is ")
print(response)
```

Usually, you will instantiate an LLM and pass it to `Settings`, which you then pass to other stages of the pipeline, as in this example:

```python
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

Settings.llm = OpenAI(temperature=0.2, model="gpt-4")

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
)
```

In this case, you've instantiated OpenAI and customized it to use the `gpt-4` model instead of the default `gpt-3.5-turbo`, and also modified the `temperature`. The `VectorStoreIndex` will now use gpt-4 to answer questions when querying.

!!! tip
    The `Settings` is a bundle of configuration data that you pass into different parts of LlamaIndex. You can [learn more about Settings](../../module_guides/supporting_modules/settings.md) and how to customize it.

## Available LLMs

We support integrations with OpenAI, Hugging Face, PaLM, and more. Check out our [module guide to LLMs](../../module_guides/models/llms.md) for a full list, including how to run a local model.

!!! tip
    A general note on privacy and LLMs can be found on the [privacy page](./privacy.md).

### Using a local LLM

LlamaIndex doesn't just support hosted LLM APIs; you can also [run a local model such as Llama2 locally](https://replicate.com/blog/run-llama-locally).

For example, if you have [Ollama](https://github.com/ollama/ollama) installed and running:

```python
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

Settings.llm = Ollama(model="llama2", request_timeout=60.0)
```

See the [custom LLM's How-To](../../module_guides/models/llms/usage_custom.md) for more details.

## Prompts

By default LlamaIndex comes with a great set of built-in, battle-tested prompts that handle the tricky work of getting a specific LLM to correctly handle and format data. This is one of the biggest benefits of using LlamaIndex. If you want to, you can [customize the prompts](../../module_guides/models/prompts/index.md)
