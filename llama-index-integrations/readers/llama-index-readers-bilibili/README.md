# Bilibili Transcript Loader

```bash
pip install llama-index-readers-bilibili
```

This loader utilizes the `bilibili_api` to fetch the text transcript from Bilibili, one of the most beloved long-form video sites in China.

With this BilibiliTranscriptReader, users can easily obtain the transcript of their desired video content on the platform.

## Usage

To use this loader, you need to pass in an array of Bilibili video links.

```python
from llama_index.readers.bilibili import BilibiliTranscriptReader

loader = BilibiliTranscriptReader()
documents = loader.load_data(
    video_urls=["https://www.bilibili.com/video/BV1yx411L73B/"]
)
```

Note that there is no official API available for Bilibili Transcript, so changes to the official website can sometimes cause issues.

This loader is designed to be used as a way to load data into [Llama Index](https://github.com/run-llama/llama_index/).
