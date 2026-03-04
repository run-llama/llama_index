# LlamaIndex Readers Integration: Obsidian

## Overview

Pass in the path to an Obsidian vault and it will parse all markdown
files into a List of Documents. Documents are split by header in
the Markdown Reader we use.

Each document will contain the following metadata:

- file_name: the name of the markdown file
- folder_path: the full path to the folder containing the file
- folder_name: the relative path to the folder containing the file
- note_name: the name of the note (without the .md extension)
- wikilinks: a list of all wikilinks found in the document
- backlinks: a list of all notes that link to this note

Optionally, tasks can be extracted from the text and stored in metadata.

### Usage

```python
from llama_index.readers.obsidian import ObsidianReader

# Initialize ObsidianReader with the path to the Obsidian vault
reader = ObsidianReader(
    input_dir="<Path to Obsidian Vault>",
    extract_tasks=False,
    remove_tasks_from_text=False,
)

# Load data from the Obsidian vault
documents = reader.load_data()
```

##### Arguments

- **input_dir** (str): Path to the Obsidian vault.
- **extract_tasks** (bool): If True, extract tasks from the text and store them in metadata. Default is False.
- **remove_tasks_from_text** (bool): If True and extract_tasks is True, remove the task lines from the main document text. Default is False.

Implementation for Obsidian reader can be found [here](https://docs.llamaindex.ai/en/stable/examples/data_connectors/ObsidianReaderDemo/)

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
