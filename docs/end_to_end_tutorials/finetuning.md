# Finetuning

## Overview

Finetuning a model means updating the model itself over a set of data to improve the model in a variety of ways. This can include improving the quality of outputs, reducing hallucinations, memorizing more data holistically, and reducing latency/cost.

The core of our toolkit revolves around in-context learning / retrieval augmentation, which involves using the models in inference mode and not training the models themselves.

While finetuning can be also used to "augment" a model with external data, finetuning can complement retrieval augmentation in a variety of ways:

#### Embedding Finetuning Benefits
- Finetuning the embedding model can allow for more meaningful embedding representations over a training distribution of data --> leads to better retrieval performance.

#### LLM Finetuning Benefits
- Allow it to learn a style over a given dataset
- Allow it to learn a DSL that might be less represented in the training data (e.g. SQL) 
- Allow it to correct hallucinations/errors that might be hard to fix through prompt engineering
- Allow it to distill a better model (e.g. GPT-4) into a simpler/cheaper model (e.g. gpt-3.5, Llama 2)


## Integrations with LlamaIndex

This is an evolving guide, and there are currently three key integrations with LlamaIndex. Please check out the sections below for more details!
- Finetuning embeddings for better retrieval performance
- Finetuning Llama 2 for better text-to-SQL
- Finetuning gpt-3.5-turbo to distill gpt-4


### Finetuning Embeddings for Better Retrieval Performance


We created a comprehensive repo/guide showing you how to finetune an open-source embedding model (in this case, `bge`) over an unstructured text corpus. It consists of the following steps:
1. Generating a synthetic question/answer dataset using LlamaIndex over any unstructed context.
2. Finetuning the model
3. Evaluating the model.

Finetuning gives you a 5-10% increase in retrieval evaluation metrics. You can then plug this fine-tuned model into your RAG application with LlamaIndex. 

```{toctree}
---
maxdepth: 1
---
Embedding Fine-tuning Guide </examples/finetuning/embeddings/finetune_embedding.ipynb>
```

**Old**
```{toctree}
---
maxdepth: 1
---
Embedding Fine-tuning Repo <https://github.com/run-llama/finetune-embedding>
Embedding Fine-tuning Blog <https://medium.com/llamaindex-blog/fine-tuning-embeddings-for-rag-with-synthetic-data-e534409a3971>
```

### Finetuning GPT-3.5 to distill GPT-4

We have multiple guides showing how to use OpenAI's finetuning endpoints to fine-tune gpt-3.5-turbo to output GPT-4 responses for RAG/agents.

We use GPT-4 to automatically generate questions from any unstructured context, and use a GPT-4 query engine pipeline to generate "ground-truth" answers. Our `OpenAIFineTuningHandler` callback automatically logs questions/answers to a dataset. 

We then launch a finetuning job, and get back a distilled model. We can evaluate this model with [Ragas](https://github.com/explodinggradients/ragas) to benchmark against a naive GPT-3.5 pipeline.

```{toctree}
---
maxdepth: 1
---
GPT-3.5 Fine-tuning Notebook (Colab) <https://colab.research.google.com/drive/1NgyCJVyrC2xcZ5lxt2frTU862v6eJHlc?usp=sharing>
GPT-3.5 Fine-tuning Notebook (Notebook link) </examples/finetuning/openai_fine_tuning.ipynb>
/examples/finetuning/react_agent/react_agent_finetune.ipynb
```

**Old**

```{toctree}
---
maxdepth: 1
---
GPT-3.5 Fine-tuning Notebook (Colab) <https://colab.research.google.com/drive/1vWeJBXdFEObuihO7Z8ui2CAYkdHQORqo?usp=sharing>
GPT-3.5 Fine-tuning Notebook (in Repo) <https://github.com/jerryjliu/llama_index/blob/main/experimental/openai_fine_tuning/openai_fine_tuning.ipynb>
```


### Finetuning Llama 2 for Better Text-to-SQL 

In this tutorial, we show you how you can finetune Llama 2 on a text-to-SQL dataset, and then use it for structured analytics against any SQL database using LlamaIndex abstractions.

The stack includes `sql-create-context` as the training dataset, OpenLLaMa as the base model, PEFT for finetuning, Modal for cloud compute, LlamaIndex for inference abstractions.

```{toctree}
---
maxdepth: 1
---
Llama 2 Text-to-SQL Fine-tuning (Repo) <https://github.com/run-llama/modal_finetune_sql>
Llama 2 Text-to-SQL Fine-tuning (Notebook) <https://github.com/run-llama/modal_finetune_sql/blob/main/tutorial.ipynb>
```

