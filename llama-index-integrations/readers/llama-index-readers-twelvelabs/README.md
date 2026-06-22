# LlamaIndex Readers Integration: TwelveLabs

```bash
pip install llama-index-readers-twelvelabs
```

The `TwelveLabsVideoReader` analyzes videos with [TwelveLabs](https://twelvelabs.io) **Pegasus** and loads the result as LlamaIndex `Document`s. Pegasus is a video-language model that performs on-the-fly analysis — it understands the video's visuals **and** its audio (its own ASR), then returns text. So a video becomes a `Document` whose text is Pegasus's analysis (e.g. a description + transcript), with no frame extraction and no separate transcription step.

Set the `TWELVELABS_API_KEY` environment variable (or pass `api_key=...`). Get a key at [playground.twelvelabs.io](https://playground.twelvelabs.io).

## Usage

```python
from llama_index.readers.twelvelabs import TwelveLabsVideoReader

reader = TwelveLabsVideoReader()  # reads TWELVELABS_API_KEY from the environment

# Analyze a publicly accessible direct video URL...
docs = reader.load_data(video_url="https://example.com/video.mp4")

# ...a local file (uploaded to TwelveLabs, <= 200 MB)...
docs = reader.load_data(video_file="/path/to/video.mp4")

# ...or a video you've already uploaded as a TwelveLabs asset:
docs = reader.load_data(asset_id="<asset-id>")

print(docs[0].text)  # Pegasus's analysis of the video
```

Pass a custom `prompt` to steer the analysis, per reader or per call:

```python
reader = TwelveLabsVideoReader(prompt="Give a verbatim timestamped transcript.")
docs = reader.load_data(
    video_url="https://example.com/talk.mp4",
    prompt="What are the three key takeaways? Cite [MM:SS] timestamps.",
)
```

### Options

- `api_key` — TwelveLabs API key (defaults to `TWELVELABS_API_KEY`).
- `model` — `pegasus1.5` (default) or `pegasus1.2`.
- `prompt` — default analysis prompt; overridable per `load_data` call.
- `temperature` — sampling temperature, 0–1 (default `0.2`).
- `max_tokens` — max output tokens per analysis (default `16384`).

Exactly one of `video_url`, `video_file`, or `asset_id` must be passed to `load_data`. `video_url` must be a publicly accessible direct video file (e.g. `https://.../clip.mp4`), not a YouTube/streaming page.
