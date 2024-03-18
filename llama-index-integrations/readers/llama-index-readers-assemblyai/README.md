# AssemblyAI Audio Transcript Loader

```bash
pip install llama-index-readers-assemblyai
```

The AssemblyAI Audio Transcript Loader allows to transcribe audio files with the [AssemblyAI API](https://www.assemblyai.com/) and loads the transcribed text into documents.

To use it, you should have the `assemblyai` python package installed, and the environment variable `ASSEMBLYAI_API_KEY` set with your API key. Alternatively, the API key can also be passed as an argument.

More info about AssemblyAI:

- [Website](https://www.assemblyai.com/)
- [Get a Free API key](https://www.assemblyai.com/dashboard/signup)
- [AssemblyAI API Docs](https://www.assemblyai.com/docs)

## Usage

The `AssemblyAIAudioTranscriptReader` needs at least the `file_path` argument. Audio files can be specified as an URL or a local file path.

```python
from llama_index.readers.assemblyai import AssemblyAIAudioTranscriptReader

audio_file = "https://storage.googleapis.com/aai-docs-samples/nbc.mp3"
# or a local file path: audio_file = "./nbc.mp3"

reader = AssemblyAIAudioTranscriptReader(file_path=audio_file)

docs = reader.load_data()
```

Note: Calling `reader.load_data()` blocks until the transcription is finished.

The transcribed text is available in the `text`:

```python
docs[0].text
# "Load time, a new president and new congressional makeup. Same old ..."
```

The `metadata` contains the full JSON response with more meta information:

```python
docs[0].metadata
# {'language_code': <LanguageCode.en_us: 'en_us'>,
#  'audio_url': 'https://storage.googleapis.com/aai-docs-samples/nbc.mp3',
#  'punctuate': True,
#  'format_text': True,
#   ...
# }
```

## Transcript Formats

You can specify the `transcript_format` argument for different formats.

Depending on the format, one or more documents are returned. These are the different `TranscriptFormat` options:

- `TEXT`: One document with the transcription text
- `SENTENCES`: Multiple documents, splits the transcription by each sentence
- `PARAGRAPHS`: Multiple documents, splits the transcription by each paragraph
- `SUBTITLES_SRT`: One document with the transcript exported in SRT subtitles format
- `SUBTITLES_VTT`: One document with the transcript exported in VTT subtitles format

```python
from llama_index.readers.assemblyai import TranscriptFormat

reader = AssemblyAIAudioTranscripReader(
    file_path="./your_file.mp3",
    transcript_format=TranscriptFormat.SENTENCES,
)

docs = reader.load_data()
```

# Transcription Config

You can also specify the `config` argument to use different audio intelligence models.

Visit the [AssemblyAI API Documentation](https://www.assemblyai.com/docs) to get an overview of all available models!

```python
import assemblyai as aai

config = aai.TranscriptionConfig(
    speaker_labels=True, auto_chapters=True, entity_detection=True
)

reader = AssemblyAIAudioTranscriptReader(
    file_path="./your_file.mp3", config=config
)
```

## Pass the API Key as argument

Next to setting the API key as environment variable `ASSEMBLYAI_API_KEY`, it is also possible to pass it as argument.

```python
reader = AssemblyAIAudioTranscriptReader(
    file_path="./your_file.mp3", api_key="YOUR_KEY"
)
```
