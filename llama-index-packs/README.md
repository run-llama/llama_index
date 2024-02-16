# LlamaPacks 📦

## Llama-Pack Usage

Llama-packs can be downloaded using the `llamaindex-cli` tool that comes with `llama-index`:

```bash
llamaindex-cli download-llamapack ZephyrQueryEnginePack --download-dir ./zephyr_pack
```

Or with the `download_llama_pack` function directly:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
LlavaCompletionPack = download_llama_pack(
    "LlavaCompletionPack", "./llava_pack"
)
```
