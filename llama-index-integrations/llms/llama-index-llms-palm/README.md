# LlamaIndex Llms Integration: Palm

## Installation

To install the required packages, run:

```bash
%pip install llama-index-llms-palm
!pip install llama-index
!pip install -q google-generativeai
```

> **Note:** If you're using Colab, the above commands will install the necessary packages. If you see a notice about updating `pip`, you can do so with:
>
> ```bash
> pip install --upgrade pip
> ```

## Setup

### Import Libraries and Configure API Key

Import the necessary libraries and configure your PaLM API key:

```python
import pprint
import google.generativeai as palm

palm_api_key = ""  # Add your API key here
palm.configure(api_key=palm_api_key)
```

### Define the Model

List and select the available models that support text generation:

```python
models = [
    m
    for m in palm.list_models()
    if "generateText" in m.supported_generation_methods
]

model = models[0].name
print(model)
```

You should see output similar to:

```
models/text-bison-001
```

### Using the PaLM LLM Abstraction

Now you can use the PaLM model to generate text. Here’s how to complete a prompt:

```python
from llama_index.llms.palm import PaLM

model = PaLM(api_key=palm_api_key)

# Example prompt
prompt = "Once upon a time in a faraway land, there lived a"
response = model.complete(prompt)
print(response)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/palm/
