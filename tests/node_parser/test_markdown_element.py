from llama_index.llms import MockLLM
from llama_index.node_parser.relational.markdown_element import (
    MarkdownElementNodeParser,
)
from llama_index.schema import Document, IndexNode, TextNode


def test_md_table_extraction() -> None:
    test_data = Document(
        text="""
# This is a test

| Year | Benefits |
| ---- | -------- |
| 2020 | 12,000   |
| 2021 | 10,000   |
| 2022 | 130,000  |


# This is another test

## Maybe a subheader

| Year | Benefits | age | customers |
| ---- | -------- | --- | --------- |
| 2020 | 12,000   | 12  | 100       |
| 2021 | 10,000   | 13  | 200       |
| 2022 | 130,000  | 14  | 300       |

        """
    )

    node_parser = MarkdownElementNodeParser(llm=MockLLM())

    nodes = node_parser.get_nodes_from_documents([test_data])
    print(f"Number of nodes: {len(nodes)}")
    for i, node in enumerate(nodes, start=0):
        print(f"Node {i}: {node}, Type: {type(node)}")
    assert len(nodes) == 6
    assert isinstance(nodes[0], TextNode)
    assert isinstance(nodes[1], IndexNode)
    assert isinstance(nodes[2], TextNode)
    assert isinstance(nodes[3], TextNode)
    assert isinstance(nodes[4], IndexNode)
    assert isinstance(nodes[5], TextNode)


def test_md_table_extraction_broken_table() -> None:
    test_data = Document(
        text="""
# This is a test

| Year | Benefits |
| ---- | -------- |
| 2020 | 12,000   | not a table |
| 2021 | 10,000   |
| 2022 | 130,000  |


# This is another test

## Maybe a subheader

| Year | Benefits | age | customers |
| ---- | -------- | --- | --------- |
| 2020 | 12,000   | 12  | 100       |
| 2021 | 10,000   | 13  | 200       |
| 2022 | 130,000  | 14  | 300       |

        """
    )

    node_parser = MarkdownElementNodeParser(llm=MockLLM())

    nodes = node_parser.get_nodes_from_documents([test_data])
    print(f"Number of nodes: {len(nodes)}")
    for i, node in enumerate(nodes, start=0):
        print(f"Node {i}: {node}, Type: {type(node)}")
    assert len(nodes) == 3
    assert isinstance(nodes[0], TextNode)
    assert isinstance(nodes[1], IndexNode)
    assert isinstance(nodes[2], TextNode)


def test_complex_md() -> None:
    test_data = Document(
        text="""
# Using LLMs

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
resp = OpenAI().complete("Paul Graham is ")
print(resp)
```

```{toctree}
---
maxdepth: 1
---
llms/usage_standalone.md
llms/usage_custom.md
```

## A Note on Tokenization

By default, LlamaIndex uses a global tokenizer for all token counting. This defaults to `cl100k` from tiktoken, which is the tokenizer to match the default LLM `gpt-3.5-turbo`.

If you change the LLM, you may need to update this tokenizer to ensure accurate token counts, chunking, and prompting.

The single requirement for a tokenizer is that it is a callable function, that takes a string, and returns a list.

You can set a global tokenizer like so:

```python
from llama_index import set_global_tokenizer

# tiktoken
import tiktoken

set_global_tokenizer(tiktoken.encoding_for_model("gpt-3.5-turbo").encode)

# huggingface
from transformers import AutoTokenizer

set_global_tokenizer(
    AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").encode
)
```

## LLM Compatibility Tracking

While LLMs are powerful, not every LLM is easy to set up. Furthermore, even with proper setup, some LLMs have trouble performing tasks that require strict instruction following.

LlamaIndex offers integrations with nearly every LLM, but it can be often unclear if the LLM will work well out of the box, or if further customization is needed.

The tables below attempt to validate the **initial** experience with various LlamaIndex features for various LLMs. These notebooks serve as a best attempt to gauge performance, as well as how much effort and tweaking is needed to get things to function properly.

Generally, paid APIs such as OpenAI or Anthropic are viewed as more reliable. However, local open-source models have been gaining popularity due to their customizability and approach to transparency.

**Contributing:** Anyone is welcome to contribute new LLMs to the documentation. Simply copy an existing notebook, setup and test your LLM, and open a PR with your results.

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
| [llama2-13b-chat](https://colab.research.google.com/drive/1S3eCZ8goKjFktF9hIakzcHqDE72g0Ggb?usp=sharing) (replicate)                 | ✅                  | ✅                  | 🛑                       | ✅       | 🛑                | 🛑          | Our ReAct prompt expects structured outputs, which llama-13b struggles at                                                                                                           |
| [llama2-70b-chat](https://colab.research.google.com/drive/1BeOuVI8StygKFTLSpZ0vGCouxar2V5UW?usp=sharing) (replicate)                 | ✅                  | ✅                  | ✅                       | ✅       | 🛑                | ⚠️          | There are still some issues with parsing structured outputs, especially with pydantic programs.                                                                                     |
| [Mistral-7B-instruct-v0.1 4bit](https://colab.research.google.com/drive/1ZAdrabTJmZ_etDp10rjij_zME2Q3umAQ?usp=sharing) (huggingface) | ✅                  | 🛑                  | 🛑                       | ⚠️       | ⚠️                | ⚠️          | Mistral seems slightly more reliable for structured outputs compared to Llama2. Likely with some prompt engineering, it may do better.                                              |
| [zephyr-7b-alpha](https://colab.research.google.com/drive/16Ygf2IyGNkb725ZqtRmFQjwWBuzFX_kl?usp=sharing) (huggingface)               | ✅                  | ✅                  | ✅                       | ✅       | ✅                | ⚠️          | Overall, `zyphyr-7b-alpha` is appears to be more reliable than other open-source models of this size. Although it still hallucinates a bit, especially as an agent.                 |
| [zephyr-7b-beta](https://colab.research.google.com/drive/1UoPcoiA5EOBghxWKWduQhChliMHxla7U?usp=sharing) (huggingface)                | ✅                  | ✅                  | ✅                       | ✅       | 🛑                | ✅          | Compared to `zyphyr-7b-alpha`, `zyphyr-7b-beta` appears to perform well as an agent however it fails for Pydantic Programs                                                          |
| [stablelm-zephyr-3b](https://colab.research.google.com/drive/1USBIOs4yUkjOcxTKBr7onjlzATE-974T?usp=sharing) (huggingface)            | ✅                  | ⚠️                  | ✅                       | 🛑       | ✅                | 🛑          | stablelm-zephyr-3b does surprisingly well, especially for structured outputs (surpassing much larger models). It struggles a bit with text-to-SQL and tool use.                     |
| [starling-lm-7b-alpha](https://colab.research.google.com/drive/1Juk073EWt2utxHZY84q_NfVT9xFwppf8?usp=sharing) (huggingface)          | ✅                  | 🛑                  | ✅                       | ⚠️       | ✅                | ✅          | starling-lm-7b-alpha does surprisingly well on agent tasks. It struggles a bit with routing, and is inconsistent with text-to-SQL.                                                  |

## Modules

We support integrations with OpenAI, Hugging Face, PaLM, and more.

```{toctree}
---
maxdepth: 2
---
llms/modules.md
```

## Further reading

```{toctree}
---
maxdepth: 1
---
/module_guides/models/embeddings.md
/module_guides/models/prompts.md
/module_guides/models/llms/local.md
Run Llama2 locally <https://replicate.com/blog/run-llama-locally>
```
"""
    )
    node_parser = MarkdownElementNodeParser(llm=MockLLM())

    nodes = node_parser.get_nodes_from_documents([test_data])
    assert len(nodes) == 7
