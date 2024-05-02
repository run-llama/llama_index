# LlamaIndex Llms Integration: `mistral.rs`

To use this integration, please install the Python `mistralrs` package:

## Installation of `mistralrs` from PyPi

0. Install Rust: https://rustup.rs/

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

1. `mistralrs` depends on the `openssl` library.

To install it on Ubuntu:

```
sudo apt install libssl-dev
sudo apt install pkg-config
```

2. Install it!

- CUDA

  `pip install mistralrs-cuda`

- Metal

  `pip install mistralrs-metal`

- Apple Accelerate

  `pip install mistralrs-accelerate`

- Intel MKL

  `pip install mistralrs-mkl`

- Without accelerators

  `pip install mistralrs`

All installations will install the `mistralrs` package. The suffix on the package installed by `pip` only controls the feature activation.

## Installation from source

Please follow the instructions [here](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/README.md).

## Usage

```python
from llama_index.llms.mistral_rs import MistralRS
from mistralrs import Which

llm = MistralRS(
    which=Which.GGUF(
        tok_model_id="mistralai/Mistral-7B-Instruct-v0.1",
        quantized_model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        quantized_filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        tokenizer_json=None,
        repeat_last_n=64,
    ),
    max_new_tokens=4096,
    context_window=1024 * 5,
)
```
