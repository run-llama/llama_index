# LlamaIndex Llms Integration: Cleanlab

## Overview

Integrate with Cleanlab's Trustworthy Language Model (TLM) APIs.

## Installation

```bash
pip install llama-index-llms-cleanlab
```

## Example

With environmental variables.

```.env
CLEANLAB_API_KEY=your_api_key
```

```python
from llama_index.llms.cleanlab import CleanlabTLM

# Initialize Cleanlab's TLM without explicitly passing the API key and base
llm = CleanlabTLM()

# Make a query to the LLM
response = llm.complete("Explain the importance of open source LLMs")

print(response)
```

Without environmental variables

```python
from llama_index.llms.cleanlab import CleanlabTLM

# Set up the CleanlabTLM's class with the required API key and quality preset
llm = CleanlabTLM(
    quality_preset="best",  # supported quality presets are: 'best','high','medium','low','base'
    api_key="your_api_key",
)

# Call the complete method with a query
response = llm.complete("Explain the importance of open source LLMs")

print(response)
```
