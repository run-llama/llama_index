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
    save_attachment=None,  # Callback function to save attachments
)
```

## Saving attachments

The `lazy_load_data` function accepts an optional `save_attachment` callback function which, if defined, is called for every attachment in the email.

Its only parameter is an `imap_tools.MailAttachment` which is described in the [official documentation](https://pypi.org/project/imap-tools/#email-attributes). It must return the path of the saved attachment as a string. Every saved attachment will be added to the Document metadata with its saved filename and the original one.

Here's a simple example of the `save_attachment` function:

```python
import imap_tools


def save_attachment(attachment: imap_tools.MailAttachment) -> str:
    with open(f"attachments/custom_filename", "wb") as f:
        f.write(attachment.payload)
    return "attachments/custom_filename"
```

---

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index).
