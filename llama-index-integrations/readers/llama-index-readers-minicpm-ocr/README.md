# MiniCPM-OCR reader

```bash
pip install llama-index-readers-minicpm-ocr
pip install paddlepaddle
```

`paddlepaddle` is the CPU layout runtime. For a GPU layout device, install
`paddlepaddle-gpu` that matches your CUDA build instead.

This reader turns PDFs and images into LlamaIndex `Document` objects with
MiniCPM-OCR 1.0. PP-DocLayoutV3 finds each block. The block is cropped and sent
to an external recognition server with the prompt for its type (`Text
Recognition:`, `Table Recognition:`, `Formula Recognition:`, and the other
labels from the SDK prompt map). The 1.3B model is not bundled. Point
`base_url` at a vLLM, SGLang, OpenAI-compatible, llama.cpp, or Ollama server.

The first call downloads PP-DocLayoutV3 into `layout_dir`
(`~/.cache/minicpm-ocr/layout` by default).

## Usage

```python
from pathlib import Path

from llama_index.readers.minicpm_ocr import MiniCPMOCRReader

reader = MiniCPMOCRReader(
    base_url="http://127.0.0.1:8000",
    model="minicpm-ocr",
    backend="vllm",
)

documents = reader.load_data(Path("scan.pdf"))
print(documents[0].metadata["page"], documents[0].text[:200])
```

One document is returned per non-empty page. Images become a single document
with `page` 1. Metadata includes `page`, `file_name`, `file_path`, and
`source`. Pass `extra_info` to add more fields.

Any constructor argument overrides the matching environment variable:

| Argument          | Environment variable          | Default                             |
| ----------------- | ----------------------------- | ----------------------------------- |
| `base_url`        | `MINICPM_OCR_BASE_URL`        | required, server root without `/v1` |
| `model`           | `MINICPM_OCR_MODEL`           | `minicpm-ocr`                       |
| `backend`         | `MINICPM_OCR_BACKEND`         | `vllm`                              |
| `api_key`         | `MINICPM_OCR_API_KEY`         | empty                               |
| `max_model_len`   | `MINICPM_OCR_MAX_MODEL_LEN`   | `40960`                             |
| `startup_timeout` | `MINICPM_OCR_STARTUP_TIMEOUT` | `60`                                |
| `request_timeout` | `MINICPM_OCR_REQUEST_TIMEOUT` | `300`                               |
| `layout_device`   | `MINICPM_OCR_LAYOUT_DEVICE`   | `cpu`                               |
| `layout_dir`      | `MINICPM_OCR_LAYOUT_DIR`      | `~/.cache/minicpm-ocr/layout`       |

`backend` is one of `vllm`, `sglang`, `openai`, `llamacpp`, or `ollama`.
`pdf_zoom` defaults to `2` (144 DPI), the same page scale the SDK uses.

```python
import os

os.environ["MINICPM_OCR_BASE_URL"] = "http://127.0.0.1:8000"
reader = MiniCPMOCRReader()
documents = reader.load_data("page.png")
```
