# RAG Local CLI Pack

This LlamaPack implements a fully local version of our [RAG CLI](https://docs.llamaindex.ai/en/stable/use_cases/q_and_a/rag_cli.html),
with Mistral (through Ollama) and [BGE-M3](https://huggingface.co/BAAI/bge-m3).

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack LocalRAGCLIPack --download-dir ./local_rag_cli_pack
```

You can then inspect the files at `./local_rag_cli_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a directory. **NOTE**: You must specify `skip_load=True` - the pack contains multiple files,
which makes it hard to load directly.

We will show you how to import the agent from these files!

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
download_llama_pack("LocalRAGCLIPack", "./local_rag_cli_pack", skip_load=True)
```

From here, you can use the pack. The most straightforward way is through the CLI. You can directly run base.py, or run the `setup_cli.sh` script.

```bash
cd local_rag_cli_pack

# option 1
python base.py rag -h

# option 2 - you may need sudo
# default name is lcli_local
sudo sh setup_cli.sh
lcli_local rag -h

```

You can also directly get modules from the pack.

```python
from local_rag_cli_pack.base import LocalRAGCLIPack

pack = LocalRAGCLIPack(
    verbose=True, llm_model_name="mistral", embed_model_name="BAAI/bge-m3"
)
# will spin up the CLI
pack.run()

# get modules
rag_cli = pack.get_modules()["rag_cli"]
rag_cli.cli()
```
