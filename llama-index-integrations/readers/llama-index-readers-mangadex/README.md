# MangaDex Loader

```bash
pip install llama-index-readers-mangadex
```

This loader fetches information from the MangaDex API, by manga title.

## Usage

```python
from llama_index.readers.mangadex import MangaDexReader

loader = MangaDexReader()
documents = loader.load_data(
    titles=["manga title 1", "manga title 2"], lang="en"
)
```

## Output

### Text

Document text is the manga title. There are alternate titles for many manga, so the canonical title will be returned, even if it is not the title that the user queried with.

### Extra Info

| Data                              | Description                                                                                        |
| --------------------------------- | -------------------------------------------------------------------------------------------------- |
| id (str)                          | MangaDex manga id                                                                                  |
| author (str)                      | Author's full name                                                                                 |
| artist (str)                      | Artist's full name                                                                                 |
| description (str)                 | Manga description                                                                                  |
| original_language (str)           | The language of the source material (before translation)                                           |
| tags (List[str])                  | Describes the manga's genre, e.g. "slice of life"                                                  |
| chapter_count (int)               | How many chapters exist in the requested language                                                  |
| latest_chapter_published_at (str) | Timestamp (YYYY-MM-DDTHH:MM:SS in timezone UTC+0) for the latest chapter in the requested language |

## Examples

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
