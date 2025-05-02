# LlamaIndex Readers Integration: Genius

```bash
pip install llama-index-readers-genius
```

This loader connects to the Genius API and loads lyrics, metadata, and album art into `Documents`.

As a prerequisite, you will need to register with [Genius API](https://genius.com/api-clients) and create an app in order to get a `client_id` and a `client_secret`. You should then set a `redirect_uri` for the app. The `redirect_uri` does not need to be functional. You should then generate an access token as an instantiator for the GeniusReader.

## Usage

Here's an example usage of the GeniusReader. It will retrieve songs that match specific lyrics. Acceptable arguments are lyrics (str): The lyric snippet you're looking for and will return List[Document]: A list of documents containing songs with those lyrics.

## GeniusReader Class Methods

### `load_artist_songs`

- **Description**: Fetches all or a specified number of songs by an artist.
- **Arguments**:
  - `artist_name` (str): The name of the artist.
  - `max_songs` (Optional[int]): Maximum number of songs to retrieve.
- **Returns**: List of `Document` objects with song lyrics.

### `load_all_artist_songs`

- **Description**: Fetches all songs of an artist and saves their lyrics.
- **Arguments**:
  - `artist_name` (str): The name of the artist.
- **Returns**: List of `Document` objects with the artist's song lyrics.

### `load_artist_songs_with_filters`

- **Description**: Loads the most or least popular song of an artist based on filters.
- **Arguments**:
  - `artist_name` (str): The artist's name.
  - `most_popular` (bool): `True` for most popular song, `False` for least popular.
  - `max_songs` (Optional[int]): Max number of songs to consider for popularity.
  - `max_pages` (int): Max number of pages to fetch.
- **Returns**: `Document` with lyrics of the selected song.

### `load_song_by_url_or_id`

- **Description**: Loads a song by its Genius URL or ID.
- **Arguments**:
  - `song_url` (Optional[str]): URL of the song on Genius.
  - `song_id` (Optional[int]): ID of the song on Genius.
- **Returns**: List of `Document` objects with the song's lyrics.

### `search_songs_by_lyrics`

- **Description**: Searches for songs by a snippet of lyrics.
- **Arguments**:
  - `lyrics` (str): Lyric snippet to search for.
- **Returns**: List of `Document` objects with songs matching the lyrics.

### `load_songs_by_tag`

- **Description**: Loads songs by a specific tag or genre.
- **Arguments**:
  - `tag` (str): Tag or genre to search for.
  - `max_songs` (Optional[int]): Max number of songs to fetch.
  - `max_pages` (int): Max number of pages to fetch.
- **Returns**: List of `Document` objects with song lyrics.

```python
from llama_index.readers.genius import GeniusReader

access_token = "your_generated_access_token"

loader = GeniusReader(access_token)
documents = loader.search_songs_by_lyrics("Imagine")
```

## Example

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).

### LlamaIndex

```python
from llama_index.core import VectorStoreIndex, download_loader

from llama_index.readers.genius import GeniusReader

access_token = "your_generated_access_token"

loader = GeniusReader(access_token)
documents = loader.search_songs_by_lyrics("Imagine")
index = VectorStoreIndex.from_documents(documents)
index.query(
    "What artists have written songs that have the lyrics imagine in them?"
)
```
