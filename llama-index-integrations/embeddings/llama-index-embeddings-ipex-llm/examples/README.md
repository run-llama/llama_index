# IpexLLMEmbedding Examples

This folder contains examples showcasing how to use LlamaIndex with `ipex-llm` Embeddings integration `llama_index.embeddings.ipex_llm.IpexLLMEmbedding` on Intel CPU and GPU.

## Installation

### On Intel CPU

Please refer to [here](https://docs.llamaindex.ai/en/stable/examples/embeddings/ipex_llm/#install-llama-index-embeddings-ipex-llm) for installation details.

### On Intel GPU

Please refer to [here](https://docs.llamaindex.ai/en/stable/examples/embeddings/ipex_llm_gpu/) for install prerequisites, `llama-index-embeddings-ipex-llm` installation, and runtime configuration.

## List of Examples

### Basic Usage Example

The example [basic.py](./basic.py) shows how to run `IpexLLMEmbedding` on Intel CPU or GPU and conduct embedding tasks such as text and query embedding. Run the example as following:

```bash
python basic.py -m <path_to_model> -d <cpu_or_xpu> -t <text_to_embed> -q <query_to_embed>
```
