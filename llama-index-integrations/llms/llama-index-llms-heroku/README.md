# Heroku Managed Inference

The `llama-index-llms-heroku` package contains LlamaIndex integrations for building applications with models on Heroku's Managed Inference platform. This integration allows you to easily connect to and use AI models deployed on Heroku's infrastructure.

## Installation

```shell
pip install llama-index-llms-heroku
```

## Setup

### 1. Create a Heroku App

First, create an app in Heroku:

```bash
heroku create $APP_NAME
```

### 2. Create and Attach AI Models

Create and attach a chat model to your app:

```bash
heroku ai:models:create -a $APP_NAME claude-3-5-haiku
```

### 3. Export Configuration Variables

Export the required configuration variables:

```bash
export INFERENCE_KEY=$(heroku config:get INFERENCE_KEY -a $APP_NAME)
export INFERENCE_MODEL_ID=$(heroku config:get INFERENCE_MODEL_ID -a $APP_NAME)
export INFERENCE_URL=$(heroku config:get INFERENCE_URL -a $APP_NAME)
```

## Usage

### Basic Usage

```python
from llama_index.llms.heroku import Heroku
from llama_index.core.llms import ChatMessage, MessageRole

# Initialize the Heroku LLM
llm = Heroku()

# Create chat messages
messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content="You are a helpful assistant."
    ),
    ChatMessage(
        role=MessageRole.USER,
        content="What are the most popular house pets in North America?",
    ),
]

# Get response
response = llm.chat(messages)
print(response)
```

### Using Environment Variables

The integration automatically reads from environment variables:

```python
import os

# Set environment variables
os.environ["INFERENCE_KEY"] = "your-inference-key"
os.environ["INFERENCE_URL"] = "https://us.inference.heroku.com"
os.environ["INFERENCE_MODEL_ID"] = "claude-3-5-haiku"

# Initialize without parameters
llm = Heroku()
```

### Using Parameters

You can also pass parameters directly:

```python
import os

llm = Heroku(
    model=os.getenv("INFERENCE_MODEL_ID", "claude-3-5-haiku"),
    api_key=os.getenv("INFERENCE_KEY", "your-inference-key"),
    inference_url=os.getenv(
        "INFERENCE_URL", "https://us.inference.heroku.com"
    ),
    max_tokens=1024,
)
```

### Text Completion

```python
# Simple text completion
response = llm.complete("Explain the importance of open source LLMs")
print(response.text)
```

## Available Models

For a complete list of available models, see the [Heroku Managed Inference documentation](https://devcenter.heroku.com/articles/heroku-inference#available-models).

## Error Handling

The integration includes proper error handling for common issues:

- Missing API key
- Invalid inference URL
- Missing model configuration

## Additional Information

For more information about Heroku Managed Inference, visit the [official documentation](https://devcenter.heroku.com/articles/heroku-inference).
