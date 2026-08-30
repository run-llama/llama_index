# Gandr Tool

This tool allows Agents to use the Gandr text-to-speech API to create audio files from text. API keys are managed at https://gandr.ai and the free tier is 50,000 tokens.

## Usage

```python
from llama_index.tools.gandr import GandrToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

speech_tool = GandrToolSpec(api_key="gnd_your-key")

agent = FunctionAgent(
    tools=speech_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)
print(
    await agent.run(
        'Create speech from the following text "Hello world" and output the file to "speech.mp3"'
    )
)
```

`text_to_speech`: Takes an input string and saves the synthesized audio to a file
`get_voices`: Lists the available Gandr voice names

The tool POSTs to `https://tts.gandr.ai/v1/audio/speech` with a `Bearer` key. The key can also be provided through the `GANDR_API_KEY` environment variable.

Details:

- Voices: `gandr-mia`, `gandr-ava`, `gandr-jenny`, `gandr-dane`, `gandr-leo`, `gandr-lewis`
- Formats: `mp3` (default), `wav`, `pcm`. `pcm` is headerless s16le mono at 24000 Hz.
- Input is capped at 2000 characters per request; the tool validates this client side.
- 23 languages are supported.
- Every render is watermarked.

This loader is designed to be used as a way to load data as a Tool in a Agent.

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-gandr/examples/gandr_speech.ipynb)
