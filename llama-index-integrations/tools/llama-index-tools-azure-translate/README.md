# Azure Translate Tool

This tool connects to a Azure account and allows an Agent to perform text translation into a variet of different languages

You will need to set up an api key and translate instance using Azure, learn more here: https://learn.microsoft.com/en-us/azure/ai-services/translator/translator-overview

For a full list of supported languages see here: https://learn.microsoft.com/en-us/azure/ai-services/translator/language-support

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-azure-speech/examples/azure_speech.ipynb)

## Usage

Here's an example usage of the AzureTranslateToolSpec.

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.azure_translate import AzureTranslateToolSpec

translate_tool = AzureTranslateToolSpec(api_key="your-key", region="eastus")

agent = FunctionAgent(
    tools=translate_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)
print(await agent.run('Say "hello world" in 5 different languages'))
```

`translate`: Translate text to a target language

This loader is designed to be used as a way to load data as a Tool in a Agent.
