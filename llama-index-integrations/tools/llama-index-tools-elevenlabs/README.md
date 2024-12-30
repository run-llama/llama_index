# Elevenlabs.io Tool

This tool allows Agents to use Elevenlabs.io text-to-speech to create audio files from text. To see more and get started, visit https://elevenlabs.io/

## Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-elevenlabs/examples/elevenlabs_speech.ipynb)

```python
from llama_index.tools.elevenlabs import ElevenLabsToolSpec
from llama_index.agent.openai import OpenAIAgent

speech_tool = ElevenLabsToolSpec(api_key="your-key")

agent = OpenAIAgent.from_tools(
    speech_tool.to_tool_list(),
    verbose=True,
)
print(
    agent.chat(
        'Create speech from the following text "Hello world!" and output the file to "speech.wav"'
    )
)
```

`text_to_speech`: Takes an input string and synthesizes audio to play on the users computer
`get_voices`: Lists the dumped Pydantic models for all available elevenlabs.io voices

This loader is designed to be used as a way to load data as a Tool in a Agent.
