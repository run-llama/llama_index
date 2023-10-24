# Large Language Models

##### FAQ

1. [How to use a custom/local embedding model?](#1-how-to-define-a-custom-llm)
2. [How to use a local hugging face embedding model?](#2-how-to-use-a-different-openai-model)
3. [How can I customize my prompt](#3-how-can-i-customize-my-prompt)
4. [Is it required to fine-tune my model?](#4-is-it-required-to-fine-tune-my-model)
5. [I want to the LLM answer in Chinese/Italian/French but only answers in English, how to proceed?](#5-i-want-to-the-llm-answer-in-chineseitalianfrench-but-only-answers-in-english-how-to-proceed)

---

##### 1. How to define a custom LLM?

You can access [Usage Custom](../../core_modules//model_modules/llms/usage_custom.md#example-using-a-custom-llm-model---advanced) to define a custom LLM.

---

##### 2. How to use a different OpenAI model?

To use a different OpenAI model you can access [Configure Model](../../examples/llm/openai.ipynb) to set your own custom model.

---

##### 3. How can I customize my prompt?

You can access [Prompts](../../core_modules/model_modules/prompts.md) to learn how to customize your prompts.

---

##### 4. Is it required to fine-tune my model?

No. there's isolated modules which might provide better results, but isn't required, you can you can use llamapidnex without needing to fine-tune the model.

---

##### 5. I want to the LLM answer in Chinese/Italian/French but only answers in English, how to proceed?

To the LLM answer in another language more accurate you can update the prompts to enforce more the output language.

```py
response = query_engine.query("Rest of your query... \nRespond in Italian")
```

Alternatively:

```py
from llama_index import LLMPredictor, ServiceContext
from llama_index.llms import OpenAI

llm_predictor = LLMPredictor(system_prompt="Always respond in Italian.")

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

query_engine = load_index_from_storage(storage_context, service_context=service_context).as_query_engine()
```

---
