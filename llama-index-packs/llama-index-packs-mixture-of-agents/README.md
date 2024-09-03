# Mixture-Of-Agents Pack

Implementation Of [Mixture-Of-Agents](https://arxiv.org/abs/2406.04692) paper from TogetherAI as LlamaPack.

Disclaimer: While the paper named the method "Mixture of Agents", agents appear to refer to LLMs themselves, not actual agentic behaviour

### Approach

The capabilities of LLMs have advanced significantly, and there is now a growing number of these models available. To maximize their potential, we need to harness the collective expertise of multiple LLMs. This is where the Mixture-of-Agents (MoA) approach comes in.

The MoA approach is a layered architecture where each layer consists of multiple LLM agents. These agents collaborate by taking the outputs of other agents in the previous layer as auxiliary information to generate their responses. This collaboration allows for the refinement and enhancement of responses, as agents build upon each other's strengths. The process can be categorized into two roles: Proposers(base LLM), who generate diverse context and perspectives, and Aggregators(reference LLMs), who synthesize these proposals into a single, high-quality output. By introducing additional aggregators and iteratively refining the responses, the MoA approach aims to maximize the collaborative potential of multiple LLMs, leading to superior outcomes.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack MixtureOfAgentsPack --download-dir ./mixture_of_agents_pack
```

You can then inspect the files at `./mixture_of_agents_pack` and use them as a template for your own project.

## Code Usage

You can use LlamaPack in the following ways:

1. Install the LlamaPack.
2. Download the LlamaPack.

### 1. Install the LlamaPack:

```bash
pip install llama-index-packs-mixture-of-agents
```

### 2. Download LlamaPack:

You can download the pack to a the `./mixture_of_agents_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
MixtureOfAgentsPack = download_llama_pack(
    "MixtureOfAgentsPack", "./mixture_of_agents_pack"
)
```

Once installed or downloaded, you can use the LlamaPack as follows:

```python
# Necessary for async operations in Jupyter notebooks
import nest_asyncio

nest_asyncio.apply()

from llama_index.llms.openai import OpenAI
from llama_index.llms.mistralai import MistralAI

# Add OPENAI_API_KEY and MISTRAL_API_KEY to your env variable

mixture_of_agents_pack = MixtureOfAgentsPack(
    llm=OpenAI(model="gpt-4"),  # Aggregator
    reference_llms=[
        OpenAI(model="gpt-3.5-turbo"),
        MistralAI(model="mistral-medium"),
    ],  # Proposers
    num_layers=3,
    temperature=0.1,
    timeout=200,  # timeout for response from workflow
)
```

From here, you can use the pack, or inspect and modify the pack in `./mixture_of_agents_pack`.

The `run()` function is a light wrapper around the proposed approach in the paper.

```python
response = mixture_of_agents_pack.run("What is LlamaIndex?")
```
