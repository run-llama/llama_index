# LlamaIndex Readers Integration: IMAP

## Overview

Simple IMAP reader allows loading emails from a given mailbox. It concatenates useful fields from each email into a single document used by LlamaIndex.

## Installation

```
pip install llama-index-readers-imap
```

## Usage

```python
from llama_index.readers.imap import ImapReader

# Initialize the server
mailbox = ImapReader(
    host="<MAIL HOST>",
    username="<MAIL USERNAME>",
    password="<MAIL PASSWORD>",
)

# Lazy load emails from the given mailbox
emails = mailbox.lazy_load_data(
    folder="INBOX",  # Customize the folder to read from
    metadata_names=[
        "uid",
        "from_values",
    ],  # Customize the metadata (date is always included). You can get the full list at https://pypi.org/project/imap-tools/#email-attributes
    search_criteria=None,  # By default all emails are read, customize the query following https://pypi.org/project/imap-tools/#search-criteria
)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index).
