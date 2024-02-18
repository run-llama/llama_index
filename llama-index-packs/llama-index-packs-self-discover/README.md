# Self-Discover LlamaPack

This LlamaPack implements [Self-Discover: Large Language Models Self-Compose Reasoning Structures](https://arxiv.org/abs/2402.03620) paper.

It has two stages for the given task:

1. STAGE-1:

   a. SELECT: Selects subset of reasoning Modules.

   b. ADAPT: Adapts selected reasoning modules to the task.

   c. IMPLEMENT: It gives reasoning structure for the task.

2. STAGE-2: Uses the generated reasoning structure for the task to generate an answer.

The implementation is inspired from the [codebase](https://github.com/catid/self-discover)

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack SelfDiscoverPack --download-dir ./self_discover_pack
```

You can then inspect the files at `./self_discover_pack` and use them as a template for your own project!

## Code Usage

```python
from llama_index.llama_pack import download_llama_pack

# download and install dependencies
SelfDiscoverPack = download_llama_pack(
    "SelfDiscoverPack", "./self_discover_pack"
)
```

From here, you can use the pack. You can import the relevant modules from the download folder (in the example below we assume it's a relative import or the directory has been added to your system path) or directly use `SelfDiscoverPack`.

```python
from self_discover_pack.base import SelfDiscoverPack

self_discover_pack = SelfRAGPack(llm=llm, verbose=True)
```

```python
self_discover_pack = SelfRAGPack(llm=llm, verbose=True)
```

The run() function serves as a concise wrapper that implements the logic outlined in the "self-discover" paper, applying it to a sample task as illustrated below.

`Emma needs to prepare 50 invitations for her upcoming birthday party. She can handwrite 10 invitations in an hour. After working for 2 hours, she takes a break for 30 minutes. If she resumes writing at the same pace, how long will it take her to complete all 50 invitations?`

```python
output = pack.run("<task>")
```
