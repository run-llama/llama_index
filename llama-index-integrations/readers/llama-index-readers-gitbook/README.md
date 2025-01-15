# LlamaIndex Readers Integration: Gitbook

## Overview

Simple Gitbook Reader allows loading data from a gitbook space. It collects & converts contents from gitbook space into documents used by LlamaIndex.

### Installation

You can install Gitbook Reader via pip:

```bash
pip install llama-index-readers-gitbook
```

### Usage

```python
from llama_index.readers.gitbook import SimpleGitbookReader

# Initialize SimpleGitbookReader
reader = SimpleGitbookReader(
    api_token="<Gitbook API Token>",  # Gitbook API Token
)

# load data from Gitbook
documents = reader.load_data(
    space_id="<Gitbook Space Id>",  # Id of the gitbook space
    metadata_names=None,  # Names of the fields to add to metadata attribute (available: 'path', 'title', 'description', 'parent')
)
```
