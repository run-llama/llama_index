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

video_ids = ["S_0hBL4ILAg", "a2skIq6hFiY"]  # Example video IDs
yt_metadata = YouTubeMetaData(api_key=api_key)
print("Testing YouTubeMetaData...")
print(yt_metadata.load_data(video_ids))

yt_meta_transcript = YouTubeMetaDataAndTranscript(api_key=api_key)
print("Testing YouTubeMetaDataAndTranscript...")
print(yt_meta_transcript.load_data(video_ids))
```

The Video_id for youtube videos is right in the URL. In this URL: https://www.youtube.com/watch?v=a2skIq6hFiY&t=60s

The video_Id is 'a2skIq6hFiY&t'.
