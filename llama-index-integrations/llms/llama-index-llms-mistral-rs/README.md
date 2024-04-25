# LlamaIndex Llms Integration: `mistral.rs`

To use this integration, please install the Python `mistralrs` package and then

## Installation of the `mistralrs` package

Please follow the simple instructions [here](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/README.md).

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
