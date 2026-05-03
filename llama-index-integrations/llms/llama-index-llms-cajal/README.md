# LlamaIndex LLM Integration — CAJAL

[![PyPI](https://img.shields.io/pypi/v/llama-index-llms-cajal)](https://pypi.org/project/llama-index-llms-cajal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/Ollama-compatible-green)](https://ollama.com)

Official LlamaIndex integration for [CAJAL](https://github.com/Agnuxo1/CAJAL) — a fine-tuned 4B-parameter model that generates publication-ready scientific papers with verified arXiv citations, running 100% locally via Ollama.

## Features

- **7-section paper generation** (Abstract → Introduction → Methodology → Results → Discussion → Conclusion → References)
- **Verified arXiv citations** — every reference is checked against the real arXiv API
- **Tribunal scoring** — optional multi-pass review with simulated peer reviewers
- **100% local inference** via Ollama — zero API costs, full data privacy
- **Streaming support** — real-time paper generation

## Installation

```bash
pip install llama-index-llms-cajal
```

Requires [Ollama](https://ollama.com) with the CAJAL model:

```bash
ollama run cajal-p2pclaw
```

## Usage

### Basic Completion

```python
from llama_index.llms.cajal import CajalLLM

llm = CajalLLM(base_url="http://localhost:11434", model="cajal-p2pclaw")
response = llm.complete("Generate a paper on quantum machine learning")
print(response.text)
```

### With LlamaIndex Settings

```python
from llama_index.core import Settings
from llama_index.llms.cajal import CajalLLM

Settings.llm = CajalLLM()

# Now use with any LlamaIndex component (RAG, agents, query engines)
```

### Streaming

```python
response = llm.stream_complete("Generate a paper on federated learning")
for chunk in response:
    print(chunk.delta, end="", flush=True)
```

### Scientific Paper Helper

```python
from llama_index.llms.cajal import generate_scientific_paper

paper = generate_scientific_paper(
    topic="Decentralized scientific peer review using blockchain",
    include_tribunal=True,  # Run simulated peer review
)
print(paper)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_url` | `http://localhost:11434` | Ollama API endpoint |
| `model` | `cajal-p2pclaw` | Model name |
| `temperature` | `0.7` | Sampling temperature |
| `max_tokens` | `4096` | Max tokens per response |
| `system_prompt` | CAJAL default | System instruction for paper generation |

## Links

- **GitHub:** https://github.com/Agnuxo1/CAJAL
- **HuggingFace:** https://huggingface.co/Agnuxo/CAJAL-4B-P2PCLAW
- **PyPI (CAJAL):** https://pypi.org/project/cajal-p2pclaw/
- **Paper:** https://arxiv.org/pdf/2604.19792

## License

MIT — same as CAJAL and LlamaIndex.
