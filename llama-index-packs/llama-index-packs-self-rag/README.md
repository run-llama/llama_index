# Simple self-RAG short form pack

This LlamaPack implements (\*in short form) the [self-RAG paper by Akari et al.](https://arxiv.org/pdf/2310.11511.pdf).

This paper presents a novel framework called Self-Reflective Retrieval-Augmented Generation (SELF-RAG). Which aims to enhance the quality and factuality of large language models (LLMs) by combining retrieval and self-reflection mechanisms.

The implementation is adapted from the author [implementation](https://github.com/AkariAsai/self-rag)
A full notebook guide can be found [here](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/self_rag/self_rag.ipynb).

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack SelfRAGPack --download-dir ./self_rag_pack
```

You can then inspect the files at `./self_rag_pack` and use them as a template for your own project!

## Code Usage

We will show you how to import the agent from these files!
The implementation uses llama-cpp, to download the relevant models (be sure to replace DIR_PATH)

```bash
pip3 install -q huggingface-hub
huggingface-cli download m4r1/selfrag_llama2_7b-GGUF selfrag_llama2_7b.q4_k_m.gguf --local-dir "<DIR_PATH>" --local-dir-use-symlinks False
```

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
SelfRAGPack = download_llama_pack("SelfRAGPack", "./self_rag_pack")
```

From here, you can use the pack. You can import the relevant modules from the download folder (in the example below we assume it's a relative import or the directory
has been added to your system path).

```python
from self_rag_pack.base import SelfRAGQueryEngine

query_engine = SelfRAGQueryEngine(
    model_path=model_path, retriever=retriever, verbose=True
)

response = query_engine.query(
    "Who won best Director in the 1972 Academy Awards?"
)
```

You can also use/initialize the pack directly.

```python
from llm_compiler_agent_pack.base import SelfRAGPack

agent_pack = SelfRAGPack(
    model_path=model_path, retriever=retriever, verbose=True
)
```

The `run()` function is a light wrapper around `agent.chat()`.

```python
response = pack.run("Who won best Director in the 1972 Academy Awards?")
```
