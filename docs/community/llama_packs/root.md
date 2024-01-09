# Llama Packs 🦙📦

## Concept

Llama Packs are a community-driven hub of **prepackaged modules/templates** you can use to kickstart your LLM app.

This directly tackles a big pain point in building LLM apps; every use case requires cobbling together custom components and a lot of tuning/dev time. Our goal is to accelerate that through a community led effort.

They can be used in two ways:

- On one hand, they are **prepackaged modules** that can be initialized with parameters and run out of the box to achieve a given use case (whether that’s a full RAG pipeline, application template, and more). You can also import submodules (e.g. LLMs, query engines) to use directly.
- On another hand, LlamaPacks are **templates** that you can inspect, modify, and use.

**All packs are found on [LlamaHub](https://llamahub.ai/).** Go to the dropdown menu and select "LlamaPacks" to filter by packs.

**Please check the README of each pack for details on how to use**. [Example pack here](https://llamahub.ai/l/llama_packs-voyage_query_engine).

See our [launch blog post](https://blog.llamaindex.ai/introducing-llama-packs-e14f453b913a) for more details.

## Usage Pattern

You can use Llama Packs through either the CLI or Python.

CLI:

```bash
llamaindex-cli download-llamapack <pack_name> --download-dir <pack_directory>
```

Python:

```python
from llama_index.llama_pack import download_llama_pack

# download and install dependencies
pack_cls = download_llama_pack("<pack_name>", "<pack_directory>")
```

You can use the pack in different ways, either to inspect modules, run it e2e, or customize the templates.

```python
# every pack is initialized with different args
pack = pack_cls(*args, **kwargs)

# get modules
modules = pack.get_modules()
display(modules)

# run (every pack will have different args)
output = pack.run(*args, **kwargs)
```

Importantly, you can/should also go into `pack_directory` to inspect the source files/customize it. That's part of the point!

## Module Guides

Some example module guides are given below. Remember, go on [LlamaHub](https://llamahub.ai) to access the full range of packs.

```{toctree}
---
maxdepth: 1
---
/examples/llama_hub/llama_packs_example.ipynb
/examples/llama_hub/llama_pack_resume.ipynb
/examples/llama_hub/llama_pack_ollama.ipynb
```
