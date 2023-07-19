# LLM

## Concept
Picking the proper Large Language Model (LLM) is one of the first steps you need to consider when building any LLM application over your data.

LLMs are a core component of LlamaIndex. They can be used as standalone modules or plugged into other core LlamaIndex modules (indices, retrievers, query engines). They are always used during the response synthesis step (e.g. after retrieval). Depending on the type of index being used, LLMs may also be used during index construction, insertion, and query traversal.

LlamaIndex provides a unified interface for defining LLM modules, whether it's from OpenAI, Hugging Face, or LangChain, so that you 
don't have to write the boilerplate code of defining the LLM interface yourself. This interface consists of the following (more details below):
- Support for **text completion** and **chat** endpoints (details below)
- Support for **streaming** and **non-streaming** endpoints
- Support for **synchronous** and **asynchronous** endpoints


## Usage Pattern

The following code snippet shows how you can get started using LLMs.

```python
from llama_index.llms import OpenAI

# non-streaming
resp = OpenAI().complete('Paul Graham is ')
print(resp)
```

You can use the LLM as a standalone module or with other LlamaIndex abstractions. Check out our guide below.

```{toctree}
---
maxdepth: 1
---
usage_standalone.md
usage_custom.md
```

To save costs, you may want to use a local model. Note that these models may be slow and may not generate high-quality outputs.

```python
from llama_index import ServiceContext
service_context = ServiceContext.from_defaults(llm_predictor="local:cpu")
```

The default uses a 4-bit quantized llama-2-13b requiring about 10GB of RAM running on CPU using [llama.cpp](https://github.com/ggerganov/llama.cpp).


## Modules

We support integrations with OpenAI, Hugging Face, PaLM, and more.

```{toctree}
---
maxdepth: 2
---
modules.md
```



