# Youtube Transcript Loader

This loader fetches the text transcript of Youtube videos using the `youtube_transcript_api` Python package.

## Usage

To use this loader, you will need to first `pip install youtube_transcript_api`.

Then, simply pass an array of YouTube links into `load_data`:

```python
from llama_hub.youtube_transcript import YoutubeTranscriptReader

loader = YoutubeTranscriptReader()
documents = loader.load_data(
    ytlinks=["https://www.youtube.com/watch?v=i3OYlaoj-BM"]
)
```

Supported URL formats: + youtube.com/watch?v={video_id} (with or without 'www.') + youtube.com/embed?v={video_id} (with or without 'www.') + youtu.be/{video_id} (never includes www subdomain)

To programmatically check if a URL is supported:

```python
from llama_hub.youtube_transcript import is_youtube_video

is_youtube_video("https://youtube.com/watch?v=j83jrh2")  # => True
is_youtube_video("https://vimeo.com/272134160")  # => False
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.
