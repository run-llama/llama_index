# 👆🏻Llama🦙🌟Tonic

Welcome🙋🏻‍♂️to the Llama-Tonic Llama Pack ! Tonic-AI is an opensource builders' community that uses `llama-index`frequently. Here we're sharing some common tools that we use ourselves with the `llama-index` community. 

<details>
<summary> ## 🧑🏽‍🤝‍👩🏼Contributing</summary>


We are thrilled you're considering contributing to Llama-Tonic! Being part of the Tonic-AI open source community means collaborating with talented builders and creators dedicated to enhancing the `llama-index` experience. Here’s how you can [join us](https://discord.gg/rAEGH3B49b) and start contributing:

### Step 1: Join Our Community

Before contributing, it’s a good idea to get familiar with our community and projects. Join our Discord server to connect with other contributors and get insights on project needs and directions. Here is the link to join: [Join Llama-Tonic Discord](https://discord.gg/rAEGH3B49b)

### Step 2: Sign Up and Set Up

Visit our GitLab repository to view the project code and issues. You will need to sign up if you haven't already:

[Sign up](https://git.tonic-ai.com) and [Explore our GitLab Repository](https://git.tonic-ai.com/contribute/LlamaIndex/LlamaTonic)

### Step 3: Open an Issue

If you notice a bug, have suggestions for improvements, or especially a new feature idea, please check the issue tracker to see if someone else has already submitted a similar issue. If not, open a new issue and clearly describe your bug, idea, or suggestion.

### Step 4: Create a Named Branch

Once your proposal is approved, or you want to tackle an existing issue, fork the repository and create a named branch from the main branch where you can work on your changes. Using a named branch helps organize reviews and integration. For example:

```bash
git checkout -b devbranch/add-mem-gpt
```

### Step 5: Build and Test

- Develop your feature contribution.
- Build tests for new codes and validate that all tests pass. 
- Document any new code with comments and update the README or associated documentation as necessary.

### Join Team Tonic

By contributing cool features to `Llama-Tonic`, you become a part of `Team Tonic`. Team Tonic and `Tonic-AI` are always building and evolving, and we are excited to see where your creativity and talent take this project!

[Let's build together and make Llama-Tonic even better](https://discord.gg/rAEGH3B49b). Thank you for your interest and we look forward to your contributions!

</details>

## Packs

- [ ] Agentic Memory
- [x] Transcription
  - [ ] Improve Results With Student-Teacher Mode

## Installation

```bash
pip install llama-index-pack-llama-tonic
```

## ✍🏻Transcription:

`./llama_tonic/transcription/whisper.py` contains a class `Transcribe`. This class is designed to perform automatic speech recognition (ASR) using the `distil-whisper/distil-large-v3` model to transcribe audio files into text. Here is a simple guide and example usage of how to utilize the `Transcribe` class for transcribing audio content.

### Why it's Cool😎: 

- **Deployable:** runs on CPU & GPU
- **Extremely Quick:** much faster than APIs
- **Precise:** <1% error rate
- **Super Easy Useage** with `llama-index`: file in , text out , the rest is handled accordingly.

### Prerequisites:

Before using the `Transcribe` class, make sure you have the necessary libraries installed. Install the required libraries using pip:

```bash
pip install llama-index-pack-llama-tonic-transcription
```

### CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack llama-tonic --download-dir ./llama_tonic
```

You can then inspect the files at `./llama_tonic` and use them as a template for your own project!

You can also use it directly in the command line:

```bash
llamaindex-cli llama-tonic-transcription --audio_file./path/to/your/audio.wav
```

### Code Description:

- **Class Initialization (`__init__`)**: The class initializes a model designed for speech-to-text transformation. It automatically selects the computing device (GPU if available; otherwise CPU) and the data type (`torch.float16` for GPU to optimize memory, and `torch.float32` for CPU).
- **Transcription Method (`transcribe`)**: This method takes the path to an audio file as input and returns the transcribed text as output. It uses a processing pipeline configured with the model.

### Programmatic Usage:
Here's how you can use the `Transcribe` class to transcribe audio files:

```python

from llama_index.packs.llama_tonic.transcription import Transcribe

def main():
    # Initialize the transcriber
    transcriber = Transcribe()
    
    # Path to your audio file
    audio_file_path = "path_to_your_audio_file.wav"
    
    # Transcribing the audio file to text
    transcribed_text = transcriber.transcribe(audio_file_path)
    
    # Print the result
    print("Transcribed Text:", transcribed_text)

if __name__ == "__main__":
    main()
```

### Notes:

- ***When using `Transcribe`for the first time , it can take a while to download and load the model for the first transcription, but the next ones are super fast !***

### Tests:
The provided code setup also includes unit tests in `tests/test_packs_llama_tonic.py` which can be run using `pytest` to ensure functionality of the transcriber. It validates basic functionality, error handling, and the configuration of the device and data types.

That's how you can integrate and use the `Transcribe` class for speech-to-text applications, harnessing the powerful ASR capability of `transformers` in Python. This allows applications ranging from automated transcription services, voice command interfaces, to more complex audio processing tasks in your `llama-index agents`.