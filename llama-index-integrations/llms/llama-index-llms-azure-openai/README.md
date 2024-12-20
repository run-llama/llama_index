# LlamaIndex Llms Integration: Azure Openai

### Installation

```bash
%pip install llama-index-llms-azure-openai
!pip install llama-index
```

### Prerequisites

Follow this to setup your Azure account: [Setup Azure account](https://docs.llamaindex.ai/en/stable/examples/llm/azure_openai/#prerequisites)

### Set the environment variables

```py
OPENAI_API_VERSION = "2023-07-01-preview"
AZURE_OPENAI_ENDPOINT = "https://YOUR_RESOURCE_NAME.openai.azure.com/"
OPENAI_API_KEY = "<your-api-key>"

import os

os.environ["OPENAI_API_KEY"] = "<your-api-key>"
os.environ[
    "AZURE_OPENAI_ENDPOINT"
] = "https://<your-resource-name>.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"

# Use your LLM
from llama_index.llms.azure_openai import AzureOpenAI

# Unlike normal OpenAI, you need to pass an engine argument in addition to model.
# The engine is the name of your model deployment you selected in Azure OpenAI Studio.

llm = AzureOpenAI(
    engine="simon-llm", model="gpt-35-turbo-16k", temperature=0.0
)

# Alternatively, you can also skip setting environment variables, and pass the parameters in directly via constructor.
llm = AzureOpenAI(
    engine="my-custom-llm",
    model="gpt-35-turbo-16k",
    temperature=0.0,
    azure_endpoint="https://<your-resource-name>.openai.azure.com/",
    api_key="<your-api-key>",
    api_version="2023-07-01-preview",
)

# Use the complete endpoint for text completion
response = llm.complete("The sky is a beautiful blue and")
print(response)

# Expected Output:
# the sun is shining brightly. Fluffy white clouds float lazily across the sky,
# creating a picturesque scene. The vibrant blue color of the sky brings a sense
# of calm and tranquility...
```

### Streaming completion

```py
response = llm.stream_complete("The sky is a beautiful blue and")
for r in response:
    print(r.delta, end="")

# Expected Output (Stream):
# the sun is shining brightly. Fluffy white clouds float lazily across the sky,
# creating a picturesque scene. The vibrant blue color of the sky brings a sense
# of calm and tranquility...

# Use the chat endpoint for conversation
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality."
    ),
    ChatMessage(role="user", content="Hello"),
]

response = llm.chat(messages)
print(response)

# Expected Output:
# assistant: Ahoy there, matey! How be ye on this fine day? I be Captain Jolly Roger,
# the most colorful pirate ye ever did lay eyes on! What brings ye to me ship?
```

### Streaming chat

```py
response = llm.stream_chat(messages)
for r in response:
    print(r.delta, end="")

# Expected Output (Stream):
# Ahoy there, matey! How be ye on this fine day? I be Captain Jolly Roger,
# the most colorful pirate ye ever did lay eyes on! What brings ye to me ship?

# Rather than adding the same parameters to each chat or completion call,
# you can set them at a per-instance level with additional_kwargs.
llm = AzureOpenAI(
    engine="simon-llm",
    model="gpt-35-turbo-16k",
    temperature=0.0,
    additional_kwargs={"user": "your_user_id"},
)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/azure_openai/
