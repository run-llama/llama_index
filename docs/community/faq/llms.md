# Large Language Models

##### FAQ

1. [How to use a custom/local embedding model?](#1-how-to-define-a-custom-llm)
2. [How to use a local hugging face embedding model?](#2-how-to-use-a-different-openai-model)
3. [How can I customize my prompt](#3-how-can-i-customize-my-prompt)
4. [Is it required to fine-tune my model?](#4-is-it-required-to-fine-tune-my-model)
5. [I want to the LLM answer in Chinese/Italian/French but only answers in English, how to proceed?](#5-i-want-to-the-llm-answer-in-chineseitalianfrench-but-only-answers-in-english-how-to-proceed)
6. [Is LlamaIndex GPU accelerated?](#6-is-llamaindex-gpu-accelerated)

---

##### 1. How to define a custom LLM?

You can access [Usage Custom](../../module_guides/models/llms/usage_custom.md#example-using-a-custom-llm-model---advanced) to define a custom LLM.

---

##### 2. How to use a different OpenAI model?

To use a different OpenAI model you can access [Configure Model](../../examples/llm/openai.ipynb) to set your own custom model.

---

##### 3. How can I customize my prompt?

You can access [Prompts](../../module_guides/models/prompts.md) to learn how to customize your prompts.

---

##### 4. Is it required to fine-tune my model?

No. there's isolated modules which might provide better results, but isn't required, you can use llamaindex without needing to fine-tune the model.

---

##### 5. I want to the LLM answer in Chinese/Italian/French but only answers in English, how to proceed?

To the LLM answer in another language more accurate you can update the prompts to enforce more the output language.

```py
response = query_engine.query("Rest of your query... \nRespond in Italian")
```

Alternatively:

```py
from llama_index import ServiceContext
from llama_index.llms import OpenAI

llm = OpenAI(system_prompt="Always respond in Italian.")

service_context = ServiceContext.from_defaults(llm=llm)

query_engine = load_index_from_storage(
    storage_context, service_context=service_context
).as_query_engine()
```

---

##### 6. Is LlamaIndex GPU accelerated?

Yes, you can run a language model (LLM) on a GPU when running it locally. You can find an example of setting up LLMs with GPU support in the [llama2 setup](../../examples/vector_stores/SimpleIndexDemoLlama-Local.ipynb) documentation.

---
