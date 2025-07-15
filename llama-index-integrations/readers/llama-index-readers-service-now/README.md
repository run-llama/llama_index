# ServiceNow Knowledge Base Loader

```bash
pip install llama-index-readers-service-now
```

This loader reads Knowledge Base articles from a ServiceNow instance. The user needs to specify the ServiceNow instance URL and authentication credentials to initialize the SnowKBReader.

The loader uses the `pysnc` library to connect to ServiceNow and retrieve knowledge base articles. It supports authentication via username/password with OAuth2 client credentials (password grant flow).

## Authentication

The reader requires the following authentication parameters:

- `instance`: Your ServiceNow instance URL (e.g., "dev12345.service-now.com")
- `username`: ServiceNow username
- `password`: ServiceNow password
- `client_id`: OAuth2 client ID (for password grant flow)
- `client_secret`: OAuth2 client secret (for password grant flow)

## Features

- Load knowledge base articles by sys_id or KB numbers
- Automatically download and process attachments
- Support for custom parsers for different file types
- Event-driven architecture with callbacks for processing
- Configurable knowledge base table (defaults to `kb_knowledge`)
- Support for filtering by workflow state (defaults to "Published")

## Supported Attachment Types

The loader can process various attachment types:

- PDF documents
- Images (PNG, JPEG, JPG, SVG)
- Microsoft Office documents (Word, Excel, PowerPoint)
- Text files (TXT, HTML, CSV, Markdown)

## Knowledge Base Article Retrieval

Articles can be retrieved in two ways:

1. `article_sys_id`: Load a specific article by its sys_id
2. `numbers`: Load articles by their KB numbers (can specify multiple numbers)

## Usage

Here's an example usage of the SnowKBReader.

```python
# Example that reads a specific KB article by sys_id
from llama_index.readers.service_now import SnowKBReader

reader = SnowKBReader(
    instance="dev12345.service-now.com",
    username="your_username",
    password="your_password",
    client_id="your_client_id",
    client_secret="your_client_secret",
)

# Load a specific article by sys_id
documents = reader.load_data(article_sys_id="your_article_sys_id")
```

```python
# Example that reads multiple KB articles by their numbers
from llama_index.readers.service_now import SnowKBReader

reader = SnowKBReader(
    instance="dev12345.service-now.com",
    username="your_username",
    password="your_password",
    client_id="your_client_id",
    client_secret="your_client_secret",
)

# Load articles by KB numbers
kb_numbers = ["KB0010001", "KB0010002", "KB0010003"]
documents = reader.load_data(numbers=kb_numbers)
```

```python
# Example with custom parsers and callbacks
from llama_index.readers.service_now import SnowKBReader, FileType
from llama_index.readers.file import PDFReader, DocxReader

# Define custom parsers for different file types
custom_parsers = {FileType.PDF: PDFReader(), FileType.DOCUMENT: DocxReader()}


# Define callback functions for processing
def process_attachment_callback(file_name: str, size: int) -> tuple[bool, str]:
    # Return (should_process, reason)
    if size > 10 * 1024 * 1024:  # Skip files larger than 10MB
        return False, "File too large"
    return True, "Processing file"


def process_document_callback(kb_number: str) -> bool:
    # Return whether to process this document
    return kb_number.startswith(
        "KB001"
    )  # Only process KB numbers starting with KB001


reader = SnowKBReader(
    instance="dev12345.service-now.com",
    username="your_username",
    password="your_password",
    client_id="your_client_id",
    client_secret="your_client_secret",
    custom_parsers=custom_parsers,
    process_attachment_callback=process_attachment_callback,
    process_document_callback=process_document_callback,
    kb_table="kb_knowledge",  # Specify custom knowledge base table
    fail_on_error=False,  # Continue processing even if some articles fail
)

documents = reader.load_data(numbers=["KB0010001", "KB0010002"])
```

```python
# Example with event monitoring
from llama_index.readers.service_now import SnowKBReader
from llama_index.readers.service_now.event import EventName


def on_page_processed(event):
    print(f"Processed page: {event.page_id}")


def on_attachment_processed(event):
    print(f"Processed attachment: {event.attachment_name}")


reader = SnowKBReader(
    instance="dev12345.service-now.com",
    username="your_username",
    password="your_password",
    client_id="your_client_id",
    client_secret="your_client_secret",
)

# Subscribe to events
reader.observer.subscribe(
    EventName.PAGE_DATA_FETCH_COMPLETED, on_page_processed
)
reader.observer.subscribe(
    EventName.ATTACHMENT_PROCESSED, on_attachment_processed
)

documents = reader.load_data(article_sys_id="your_article_sys_id")
```

## Prerequisites

Before using this loader, make sure to install the required dependencies:

```bash
pip install pysnc
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
