# LlamaIndex Readers Integration: Youtube-Metadata

```bash
pip install llama_index.readers.youtube_metadata
```

This loader fetches the metadata of Youtube videos using the Google APIs. (https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={videos_string}&key={api_key}). You must have a Google API key to use.

Transcripts of the text transcript of Youtube videos is fetched using the `youtube_transcript_api` Python package.

## Usage

Simply pass an array of YouTube Video_ID into `load_data`.

```python
from llama_index.readers.youtube_metadata import YoutubeMetaData

api_key = "Axxxxx"  # youtube API Key

video_ids = ["S_0hBL4ILAg", "a2skIq6hFiY"]

youtube_meta = YoutubeMetaData(api_key)
details = youtube_meta.load_data(video_ids)
```

This can be combined with the YoutubeTranscriptReader to provide more information for RAG AI inquiries.

```python
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.readers.youtube_metadata import YoutubeMetaData

youtube_meta = YoutubeMetaData(api_key)
details = youtube_meta.load_data(video_ids)

loader = YoutubeTranscriptReader()
documents = loader.load_data(
    ytlinks=["https://www.youtube.com/watch?v=i3OYlaoj-BM"]
)
```

Supported URL formats: + youtube.com/watch?v={video_id} (with or without 'www.') + youtube.com/embed?v={video_id} (with or without 'www.') + youtu.be/{video_id} (never includes www subdomain)
