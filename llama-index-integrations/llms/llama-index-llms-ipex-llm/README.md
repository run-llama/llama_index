# LlamaIndex Llms Integration: IPEX-LLM

[IPEX-LLM](https://github.com/intel-analytics/ipex-llm) is a PyTorch library for running LLM on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max) with very low latency. This module enables the use of LLMs optimized with `ipex-llm` in LlamaIndex pipelines.

## Installation

### On CPU

```bash
pip install llama-index-llms-ipex-llm
```

### On GPU

```bash
pip install llama-index-llms-ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

## Usage

```python
from llama_index.llms.ipex_llm import IpexLLM
```

## Examples

- [Notebook Example](https://docs.llamaindex.ai/en/stable/examples/llm/ipex_llm/)
- [More Examples](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/llms/llama-index-llms-ipex-llm/examples)
