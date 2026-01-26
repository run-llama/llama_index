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

`text_to_speech`: Convert text to speech with emotion, pitch, tempo control, and reproducible results
`get_voices`: List all available Typecast voices
`get_voice`: Get details of a specific voice by ID

This tool is designed to be used as a Tool in an Agent.

## Features

- **Multiple Voice Models**: Support for various AI voice models (ssfm-v21, ssfm-v30)
- **Multi-language Support**: 27+ languages including English, Korean, Spanish, Japanese, Chinese, and more
- **Emotion Control**: Adjust emotional expression (happy, sad, angry, normal, whisper, etc.) with intensity control
- **Audio Customization**: Control volume, pitch, tempo, and output format (WAV/MP3)
- **Reproducible Results**: Use seed parameter for consistent audio generation
- **Voice Discovery**: List and search available voices by model, gender, age, or use case (V2 API)

## Advanced Usage

### Using Seed for Reproducible Results

```python
from llama_index.tools.typecast import TypecastToolSpec

speech_tool = TypecastToolSpec(api_key="your-key")

# Generate the same audio multiple times with the same seed
result = speech_tool.text_to_speech(
    text="Hello world!",
    voice_id="tc_62a8975e695ad26f7fb514d1",
    output_path="speech.wav",
    seed=42  # Same seed = same audio
)
```

### Getting Voice Details (V2 API)

```python
# Get specific voice information
voice = speech_tool.get_voice("tc_62a8975e695ad26f7fb514d1")
print(f"Voice: {voice['voice_name']}")
print(f"Gender: {voice['gender']}, Age: {voice['age']}")
print(f"Use cases: {voice['use_cases']}")

# Models now include emotions per model version
for model in voice['models']:
    print(f"Model {model['version']}: emotions = {model['emotions']}")
```

### Filtering Voices (V2 API)

```python
# Filter by model, gender, age, and use case
voices = speech_tool.get_voices(
    model="ssfm-v30",
    gender="female",
    age="young_adult",
    use_case="Audiobook"
)

for voice in voices:
    print(f"{voice['voice_name']} ({voice['voice_id']})")
```
