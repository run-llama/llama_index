# IpexLLM Examples

This folder contains examples showcasing how to use LlamaIndex with `ipex-llm` LLM integration `llama_index.llms.ipex_llm.IpexLLM`.

## Installation

### On CPU

Install `llama-index-llms-ipex-llm`. This will also install `ipex-llm` and its dependencies.

```bash
pip install llama-index-llms-ipex-llm
```

### On GPU

Install `llama-index-llms-ipex-llm`. This will also install `ipex-llm` and its dependencies.

```bash
pip install llama-index-llms-ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

## List of Examples

### Basic Example

The example [basic.py](./basic.py) shows how to run `IpexLLM` on Intel CPU or GPU and conduct tasks such as text completion. Run the example as following:

```bash
python basic.py -m <path_to_model> -d <cpu_or_xpu_or_xpu:device_id> -q <query_to_LLM>
```

> Please note that in this example we'll use [HuggingFaceH4/zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha) model for demonstration. It requires updating `transformers` and `tokenizers` packages.
>
> ```bash
> pip install -U transformers==4.37.0 tokenizers==0.15.2
> ```

### Low Bit Example

The example [low_bit.py](./low_bit.py) shows how to save and load low_bit model by `IpexLLM` on Intel CPU or GPU and conduct tasks such as text completion. Run the example as following:

```bash
python low_bit.py -m <path_to_model> -d <cpu_or_xpu_or_xpu:device_id> -q <query_to_LLM> -s <save_low_bit_dir>
```

> Please note that in this example we'll use [HuggingFaceH4/zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha) model for demonstration. It requires updating `transformers` and `tokenizers` packages.
>
> ```bash
> pip install -U transformers==4.37.0 tokenizers==0.15.2
> ```

### More Data Types Example

By default, `IpexLLM` loads the model in int4 format. To load a model in different data formats like `sym_int5`, `sym_int8`, etc., you can use the `load_in_low_bit` option in `IpexLLM`.

```bash
python more_data_type.py -m <path_to_model> -t <path_to_tokenizer> -l <low_bit_format> -d <cpu_or_xpu_or_xpu:device_id> -q <query_to_LLM>
```

> Note: If you're using [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) model in this example, it is recommended to use transformers version
> <=4.34.
