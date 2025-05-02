# LlamaIndex Llms Integration: Huggingface

## Installation

1. Install the required Python packages:

   ```bash
   %pip install llama-index-llms-huggingface
   %pip install llama-index-llms-huggingface-api
   !pip install "transformers[torch]" "huggingface_hub[inference]"
   !pip install llama-index
   ```

2. Set the Hugging Face API token as an environment variable:

   ```bash
   export HUGGING_FACE_TOKEN=your_token_here
   ```

## Usage

### Import Required Libraries

```python
import os
from typing import List, Optional
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
```

### Run a Model Locally

To run the model locally on your machine:

```python
locally_run = HuggingFaceLLM(model_name="HuggingFaceH4/zephyr-7b-alpha")
```

### Run a Model Remotely

To run the model remotely using Hugging Face's Inference API:

```python
HF_TOKEN: Optional[str] = os.getenv("HUGGING_FACE_TOKEN")
remotely_run = HuggingFaceInferenceAPI(
    model_name="HuggingFaceH4/zephyr-7b-alpha", token=HF_TOKEN
)
```

### Anonymous Remote Execution

You can also use the Inference API anonymously without providing a token:

```python
remotely_run_anon = HuggingFaceInferenceAPI(
    model_name="HuggingFaceH4/zephyr-7b-alpha"
)
```

### Use Recommended Model

If you do not provide a model name, Hugging Face's recommended model is used:

```python
remotely_run_recommended = HuggingFaceInferenceAPI(token=HF_TOKEN)
```

### Generate Text Completion

To generate a text completion using the remote model:

```python
completion_response = remotely_run_recommended.complete("To infinity, and")
print(completion_response)
```

### Set Global Tokenizer

If you modify the LLM, ensure you change the global tokenizer to match:

```python
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer

set_global_tokenizer(
    AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha").encode
)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/huggingface/
