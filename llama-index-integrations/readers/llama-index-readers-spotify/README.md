# Spotify Loader

```bash
pip install llama-index-readers-spotify
```

This loader reads your Spotify account and loads saved albums, tracks, or playlists into `Documents`.

As a prerequisite, you will need to register with [Spotify for Developers](https://developer.spotify.com) and create an app in order to get a `client_id` and a `client_secret`. You should then set a `redirect_uri` for the app (in the web dashboard under app settings). The `redirect_uri` does not need to be functional. You should then set the `client_id`, `client_secret`, and `redirect_uri` as environmental variables.

`export SPOTIPY_CLIENT_ID='xxxxxxxxxxxxxxxxx'`\
`export SPOTIPY_CLIENT_SECRET='xxxxxxxxxxxxxxxxxx'`\
`export SPOTIPY_REDIRECT_URI='http://localhost:8080/redirect'`

## Usage

Here's an example usage of the SpotifyReader. It will retrieve your saved albums, unless an optional `collection` argument is passed. Acceptable arguments are "albums", "tracks", and "playlists".

```python
from llama_index.readers.spotify import SpotifyReader

loader = SpotifyReader()
documents = loader.load_data()
```

## Example

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).

### LlamaIndex

```python
from llama_index.core import VectorStoreIndex, download_loader

from llama_index.readers.spotify import SpotifyReader

loader = SpotifyReader()
documents = loader.load_data()
index = VectorStoreIndex.from_documents(documents)
index.query(
    "When are some other artists i might like based on what i listen to ?"
)
```
