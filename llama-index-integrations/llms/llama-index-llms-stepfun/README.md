# LlamaIndex LLM Integration: Stepfun

## Installation

To install the required packages, run:

```bash
%pip install llama-index-llms-stepfun
!pip install llama-index
```

## Setup

### Initialize StepFun LLM

First, import the necessary libraries and set up your `StepFun` instance. Replace `step-1v-8k`, and `TOKEN` with your model name, and API key, respectively:

```python
import os
from typing import List, Optional
from llama_index.llms.stepfun import StepFun
from llama_index.core.llms import ChatMessage

llm = StepFun(
    api_key="TOKEN",
    max_tokens=256,
    context_window=4096,
    model="step-1v-8k",
)
```

## Chat Functionality

StepFun supports chat APIs, allowing you to handle conversation-like interactions. Here’s how to use it:

```python
from llama_index.llms.stepfun import StepFun
from llama_index.core.llms import ChatMessage

llm = StepFun(
    api_key="",
    max_tokens=256,
    context_window=4096,
    model="step-1v-8k",
)


message = ChatMessage(role="user", content="Tell me a joke")
resp = llm.chat([message])
print(resp)
```
