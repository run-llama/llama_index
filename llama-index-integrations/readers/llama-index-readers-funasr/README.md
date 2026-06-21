# FunASR Reader

## Overview

FunASR Reader reads audio files and transcribes them to text locally with [FunASR](https://github.com/modelscope/FunASR) (SenseVoice / Paraformer / Fun-ASR-Nano). It is multilingual (Chinese, Cantonese, English, Japanese, Korean and more), runs on CPU or GPU, and needs **no API key**. SenseVoice (the default) auto-detects the spoken language, and a built-in FSMN-VAD handles long audio.

### Installation

```bash
pip install llama-index-readers-funasr
```

## Usage

```python
from llama_index.readers.funasr import FunASRReader

# Initialize FunASRReader (downloads the model on first use)
reader = FunASRReader(model="iic/SenseVoiceSmall", device="cpu")

# Load data from an audio file
documents = reader.load_data("path/to/your/audio/file.wav")
```

This loader is designed to ingest audio (meetings, podcasts, voice notes) into a LlamaIndex pipeline for retrieval and question answering.
