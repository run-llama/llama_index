# LlamaIndex Llms Integration: Mymagic

## Installation

To install the required package, run:

```bash
%pip install llama-index-llms-mymagic
!pip install llama-index
```

## Setup

Before you begin, set up your cloud storage bucket and grant MyMagic API secure access. For detailed instructions, visit the [MyMagic documentation](https://docs.mymagic.ai).

### Initialize MyMagicAI

Create an instance of MyMagicAI by providing your API key and storage configuration:

```python
from llama_index.llms.mymagic import MyMagicAI

llm = MyMagicAI(
    api_key="your-api-key",
    storage_provider="s3",  # Options: 's3' or 'gcs'
    bucket_name="your-bucket-name",
    session="your-session-name",  # Directory for batch inference
    role_arn="your-role-arn",
    system_prompt="your-system-prompt",
    region="your-bucket-region",
    return_output=False,  # Set to True to return output JSON
    input_json_file=None,  # Input file stored on the bucket
    list_inputs=None,  # List of inputs for small batch
    structured_output=None,  # JSON schema of the output
)
```

> **Note:** If `return_output` is set to `True`, `max_tokens` should be at least 100.

### Generate Completions

To generate a text completion for a question, use the `complete` method:

```python
resp = llm.complete(
    question="your-question",
    model="choose-model",  # Supported models: mistral7b, llama7b, mixtral8x7b, codellama70b, llama70b, etc.
    max_tokens=5,  # Number of tokens to generate (default is 10)
)
print(
    resp
)  # The response indicates if the final output is stored in your bucket or raises an exception if the job failed
```

### Asynchronous Requests

For asynchronous operations, use the `acomplete` endpoint:

```python
import asyncio


async def main():
    response = await llm.acomplete(
        question="your-question",
        model="choose-model",  # Supported models listed in the documentation
        max_tokens=5,  # Number of tokens to generate (default is 10)
    )
    print("Async completion response:", response)


await main()
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/mymagic/
