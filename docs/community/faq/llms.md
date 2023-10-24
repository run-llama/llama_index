# Large Language Models

---

##### Q1: How to define a custom LLM?

A1: You can access [Usage Custom](../../core_modules//model_modules/llms/usage_custom.md#example-using-a-custom-llm-model---advanced) to define a custom LLM.

---

##### Q2: How to use a different OpenAI model?

A2: To use a different OpenAI model you can access [Configure Model](../../examples/llm/openai.ipynb) to set your own custom model

---

##### Q3: How can I customize my prompt?

A3: You can access [Prompts](../../core_modules/model_modules/prompts.md) to learn how to customize your prompts.

---

##### Q4: Is it required to fine-tune my model?

A4: No. there's isolated modules which might provide better results, but isn't required, you can you can use llamapidnex without needing to fine-tune the model.

---

##### Q5: I want to the LLM answer in Chinese/Italian/French but only answers in English, how to proceed?

A5: To the LLM answer in another language more accurate you can update the prompts to enforce more the output language

```py
response = query_engine.query(input_text = "\nRespond in Italian")
```

Alternatively

```py
from llama_index import LLMPredictor, ServiceContext
from llama_index.llms import OpenAI

llm_predictor = LLMPredictor(system_prompt="Always respond in Italian.")

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

query_engine = load_index_from_storage(storage_context, service_context=service_context).as_query_engine()
```

---
