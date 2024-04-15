# IpexLLM Examples

This folder contains examples showcasing how to use LlamaIndex with `ipex-llm` LLM integration `llama_index.llms.ipex_llm.IpexLLM`.

## Installation

### On CPU

Install `llama-index-llms-ipex-llm`. This will also install `ipex-llm` and its dependencies.

```bash
pip install llama-index-llms-ipex-llm
```

## List of Examples

### More Data Types Example

By default, `IpexLLM` loads the model in int4 format. To load a model in different data formats like `sym_int5`, `sym_int8`, etc., you can use the `load_in_low_bit` option in `IpexLLM`.

The example [more_data_type.py](./more_data_type.py) shows how to use the `load_in_low_bit` option. Run the example as following:

```bash
python more_data_type.py -m <path_to_model> -t <path_to_tokenizer> -l <low_bit_format>
```

> Note: If you're using [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) model in this example, it is recommended to use transformers version
> <=4.34.
