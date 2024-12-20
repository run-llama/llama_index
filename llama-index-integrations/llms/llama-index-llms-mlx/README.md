# LlamaIndex Llms Integration: MLX

## Overview

---

Integrate with MLX LLMs from the mlx-lm library

## Installation

---

```bash
pip install llama-index-llms-mlx
```

## Example

---

```python
from llama_index.llms.mlx import MLXLLM

llm = MLXLLM(
    model_name="microsoft/phi-2",
    context_window=3900,
    max_new_tokens=256,
    generate_kwargs={"temp": 0.7, "top_p": 0.95},
)

response = llm.complete("What is the meaning of life?")
print(str(response))
```
