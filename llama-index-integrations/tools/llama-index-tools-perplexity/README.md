# Perplexity API Tool for LlamaIndex

Perplexity's Sonar API is a lightweight, fast, and affordable generative search solution that delivers real-time, web-connected research with factual answers and citations—ideal for integrating question-and-answer features into your applications. 

This tool provides a simple interface to interact with the Perplexity API (using the Sonar model) within the LlamaIndex framework. It lets you search the web with customizable parameters.

For more details on the Perplexity API and Sonar model, please refer to the official [Perplexity Documentation](https://docs.perplexity.ai/home).

## Getting started:

To use this tool, you will need a Perplexity API key. You can obtain one easily by following the steps outlined [here](https://docs.perplexity.ai/guides/getting-started). 

## Usage Example

	1.	Create a `.env` file in your project root with your API keys:

```bash
OPENAI_API_KEY=your-openai-api-key
PPLX_API_KEY=your-perplexity-api-key
```


	2.	Use the tool in your code:

```python
from dotenv import load_dotenv
import os
from llama_index.tools.perplexity.base import PerplexityToolSpec

# Load environment variables from .env
load_dotenv()

# Initialize the Perplexity tool with the API key from the environment
perplexity_tool = PerplexityToolSpec(api_key=os.getenv("PPLX_API_KEY"))

# Call chat_completion with a custom query and specify the model as "sonar-pro"
response = perplexity_tool.chat_completion("What is going on in the world today?", model="sonar-pro")
print(response)
```
