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

```{toctree}
---
maxdepth: 1
---
usage_standalone.md
usage_custom.md
```

## LLM Compatibility Tracking

While LLMs are powerful, not every LLM is easy to set up. Furthermore, even with proper setup, some LLMs have trouble performning tasks that require strict instruction following.

LlamaIndex offers integrations with nearly every LLM, but it can be often unclear if the LLM will work well out of the box, or if further customization is needed.

The tables below attempt to validate the **initial** experience with various LlamaIndex features for various LLMs. These notebooks serve as a best attempt to gauge performance, as well as how much effort and tweaking is needed to get things to function properly.

Generally, paid APIs such as OpenAI or Anthropic are viewed as more reliable. However, local open-source models have been gaining popularity due to their customizability and approach to transparency.

**Contributing:** Anyone is welcome to contribute new LLMs to the documentation. Simply copy an existing notebook, setup and test your LLM, and open a PR with your resutls.

If you have ways to improve the setup for existing notebooks, contributions to change this are welcome!

**Legend**

- ✅ = should work fine
- ⚠️ = sometimes unreliable, may need prompt engineering to improve
- 🛑 = usually unreliable, would need prompt engineering/fine-tuning to improve

### Paid LLM APIs

| Model Name                                                                                                               | Basic Query Engines | Router Query Engine | Sub Question Query Engine | Text2SQL | Pydantic Programs | Data Agents | <div style="width:290px">Notes</div>    |
| ------------------------------------------------------------------------------------------------------------------------ | ------------------- | ------------------- | ------------------------- | -------- | ----------------- | ----------- | --------------------------------------- |
| [gpt-3.5-turbo](https://colab.research.google.com/drive/1oVqUAkn0GCBG5OCs3oMUPlNQDdpDTH_c?usp=sharing) (openai)          | ✅                  | ✅                  | ✅                        | ✅       | ✅                | ✅          |                                         |
| [gpt-3.5-turbo-instruct](https://colab.research.google.com/drive/1DrVdx-VZ3dXwkwUVZQpacJRgX7sOa4ow?usp=sharing) (openai) | ✅                  | ✅                  | ✅                        | ✅       | ✅                | ⚠️          | Tool usage in data-agents seems flakey. |
| [gpt-4](https://colab.research.google.com/drive/1RsBoT96esj1uDID-QE8xLrOboyHKp65L?usp=sharing) (openai)                  | ✅                  | ✅                  | ✅                        | ✅       | ✅                | ✅          |                                         |
| [claude-2](https://colab.research.google.com/drive/1os4BuDS3KcI8FCcUM_2cJma7oI2PGN7N?usp=sharing) (anthropic)            | ✅                  | ✅                  | ✅                        | ✅       | ✅                | ⚠️          | Prone to hallucinating tool inputs.     |
| [claude-instant-1.2](https://colab.research.google.com/drive/1wt3Rt2OWBbqyeRYdiLfmB0_OIUOGit_D?usp=sharing) (anthropic)  | ✅                  | ✅                  | ✅                        | ✅       | ✅                | ⚠️          | Prone to hallucinating tool inputs.     |

### Open Source LLMs

Since open source LLMs require large amounts of resources, the quantization is reported. Quantization is just a method for reducing the size of an LLM by shrinking the accuracy of calculations within the model. Research has shown that up to 4Bit quantization can be achieved for large LLMs without impacting performance too severely.

| Model Name                                                                                                                           | Basic Query Engines | Router Query Engine | SubQuestion Query Engine | Text2SQL | Pydantic Programs | Data Agents | <div style="width:290px">Notes</div>                                                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------ | ------------------- | ------------------- | ------------------------ | -------- | ----------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [llama2-chat-7b 4bit](https://colab.research.google.com/drive/14N-hmJ87wZsFqHktrw40OU6sVcsiSzlQ?usp=sharing) (huggingface)           | ✅                  | 🛑                  | 🛑                       | 🛑       | 🛑                | ⚠️          | Llama2 seems to be quite chatty, which makes parsing structured outputs difficult. Fine-tuning and prompt engineering likely required for better performance on structured outputs. |
| [Mistral-7B-instruct-v0.1 4bit](https://colab.research.google.com/drive/1ZAdrabTJmZ_etDp10rjij_zME2Q3umAQ?usp=sharing) (huggingface) | ✅                  | 🛑                  | 🛑                       | ⚠️       | ⚠️                | ⚠️          | Mistral seems slightly more reliable for structured outputs compared to Llama2. Likely with some prompt engineering, it may do better.                                              |
| [zephyr-7b-alpha](https://colab.research.google.com/drive/16Ygf2IyGNkb725ZqtRmFQjwWBuzFX_kl?usp=sharing) (huggingface)               | ✅                  | ✅                  | ✅                       | ✅       | ✅                | ⚠️          | Overall, `zyphyr-7b` is appears to be more reliable than other open-source models of this size. Although it still hallucinates a bit, especially as an agent.                       |

## Modules

We support integrations with OpenAI, Hugging Face, PaLM, and more.

```{toctree}
---
maxdepth: 2
---
modules.md
```
