# LlamaIndex Readers Integration: FunASR

Transcribe audio into LlamaIndex `Document`s with [FunASR](https://github.com/modelscope/FunASR) — self-hosted speech-to-text powered by [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) / Paraformer / Fun-ASR-Nano. Runs locally, no cloud API; strong on Chinese and 50+ languages.

## Installation

```bash
pip install llama-index-readers-funasr
```

## Usage

```python
from llama_index.readers.funasr import FunASRReader

reader = FunASRReader(model="iic/SenseVoiceSmall", device="cuda")
documents = reader.load_data("meeting.wav")
print(documents[0].text)
```

Pass `hub="hf"` with `model="FunAudioLLM/SenseVoiceSmall"` to load from HuggingFace. The built-in VAD handles audio of any length.
