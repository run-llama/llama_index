# Project Gutenberg loader

```bash
pip install llama-index-readers-gutenberg
```

This loader loads books from their names from the Project Gutenberg library.

## Usage

Here's an example usage of the Project Gutenberg loader:

```python
from llama_index.readers.gutenberg import ProjectGutenbergReader

reader = ProjectGutenbergReader()
crime_and_punishment = reader.load_data("Crime and Punishment")

# Not existing book raises exception
reader.load_data("Not existing book")
```
