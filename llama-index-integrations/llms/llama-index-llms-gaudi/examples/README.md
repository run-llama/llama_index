# GaudiLLM Examples

This folder contains examples showcasing how to use LlamaIndex with `gaudi` LLM integration `llama_index.llms.gaudi.GaudiLLM`.

## Installation

### On Intel Gaudi

Install `llama-index-llms-gaudi`. This will also install `gaudi` and its dependencies.

```bash
pip install --upgrade-strategy eager optimum[habana]
```

## List of Examples

### Basic Example

The example [basic.py](./basic.py) shows how to run `GaudiLLM` on Intel Gaudi and conduct tasks such as text completion. Run the example as following:

```bash
python basic.py
```

> Please note that in this example we'll use [HuggingFaceH4/zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha) model for demonstration. It requires `transformers` and `tokenizers` packages.
>
> ```bash
> pip install -U transformers tokenizers
> ```
