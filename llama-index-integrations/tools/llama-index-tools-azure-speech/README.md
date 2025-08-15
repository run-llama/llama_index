# Azure Speech Tool

This tool allows Agents to use Microsoft Azure speech services to transcribe audio files to text, and create audio files from text. To see more and get started, visit https://azure.microsoft.com/en-us/products/ai-services/ai-speech

## Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-azure-speech/examples/azure_speech.ipynb)

```python
from llama_index.tools.azure_speech import AzureSpeechToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

speech_tool = AzureSpeechToolSpec(speech_key="your-key", region="eastus")

agent = FunctionAgent(
    tools=speech_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)
print(await agent.run('Say "hello world"'))
print(
    await agent.run(
        "summarize the data/speech.wav audio file into a few sentences"
    )
)
```

`text_to_speech`: Takes an input string and synthesizes audio to play on the users computer
`speech_to_text`: Takes a .wav file and transcribes it into text

This loader is designed to be used as a way to load data as a Tool in a Agent.
