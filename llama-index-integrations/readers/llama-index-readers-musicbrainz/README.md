# MusicBrainz Loader

```bash
pip install llama-index-readers-musicbrainz
```

This loader reads structured music metadata from the open
[MusicBrainz](https://musicbrainz.org) database — artists, release groups
(albums), releases, recordings (tracks), works, and labels — and loads
each entity into a LlamaIndex `Document`.

MusicBrainz is free, requires no API key, and is rate-limited to **one
request per second per User-Agent**. The reader sets a polite default
User-Agent; downstream applications should supply their own identifying
string per the MusicBrainz API guidelines.

## Usage

### Search

Searches return up to `limit` hits as Documents. The Document text is a
short human-readable summary (suitable for embedding); the full
MusicBrainz JSON payload (MBID, scores, tags, relationships) is preserved
in `Document.metadata`.

```python
from llama_index.readers.musicbrainz import MusicBrainzReader

reader = MusicBrainzReader(
    user_agent="my-app/1.0 (contact@example.com)"
)

# Search artists
artists = reader.load_data(query="Radiohead", entity="artist")

# Search release groups (albums)
albums = reader.load_data(query="OK Computer", entity="release-group", limit=5)

# Search recordings (tracks)
tracks = reader.load_data(query="Paranoid Android", entity="recording")
```

### Lookup by MBID

To hydrate sub-resources (relationships, tags, ratings, etc.), pass the
MusicBrainz Identifier and the desired `includes`:

```python
artist_docs = reader.load_data(
    mbid="a74b1b7f-71a5-4011-9441-d0b5e4122711",  # Radiohead
    entity="artist",
    includes=["release-groups", "tags", "ratings"],
)
```

Valid `entity` values:

| Entity | Description |
|---|---|
| `artist` | An individual artist, band, or other performer |
| `release-group` | A logical release such as an album (across reissues) |
| `release` | A specific edition / pressing of a release group |
| `recording` | A track / performance (may appear on multiple releases) |
| `work` | A composition (the abstract piece, independent of recording) |
| `label` | A record label |

See the [MusicBrainz API documentation](https://musicbrainz.org/doc/MusicBrainz_API)
for the full list of valid `includes` per entity type.

## Example

This loader can be combined with LlamaIndex to build a music-knowledge
retrieval system over MusicBrainz metadata:

```python
from llama_index.core import VectorStoreIndex
from llama_index.readers.musicbrainz import MusicBrainzReader

reader = MusicBrainzReader(user_agent="my-app/1.0 (you@example.com)")
documents = reader.load_data(query="Radiohead", entity="release-group", limit=50)

index = VectorStoreIndex.from_documents(documents)
response = index.as_query_engine().query(
    "What is Radiohead's earliest studio album?"
)
print(response)
```
