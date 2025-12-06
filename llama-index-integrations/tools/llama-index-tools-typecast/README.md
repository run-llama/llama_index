# Typecast.ai Tool

This tool allows Agents to use Typecast.ai text-to-speech to create audio files from text with emotion control. To see more and get started, visit https://typecast.ai/

## Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-typecast/examples/typecast_speech.ipynb)

```python
from llama_index.tools.typecast import TypecastToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

speech_tool = TypecastToolSpec(api_key="your-key")

agent = FunctionAgent(
    tools=speech_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4o-mini"),
)
print(
    await agent.run(
        'Create speech from the text "Hello world!" with a happy emotion and output the file to "speech.wav"'
    )
)
```

`text_to_speech`: Convert text to speech with emotion, pitch, and tempo control
`get_voices`: List all available Typecast voices

This tool is designed to be used as a Tool in an Agent.
