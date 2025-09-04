# ServiceNow Knowledge Base Loader

```bash
pip install llama-index-readers-service-now
```

This loader reads Knowledge Base articles from a ServiceNow instance. The user needs to specify the ServiceNow instance URL and authentication credentials to initialize the SnowKBReader.

The loader uses the `pysnc` library to connect to ServiceNow and retrieve knowledge base articles. It supports authentication via username/password (basic auth) or with OAuth2 client credentials (password grant flow).

**Important**: This reader requires custom parsers for processing different file types. At minimum, an HTML parser must be provided for processing article bodies.

## Authentication

The reader requires the following authentication parameters:

**Required:**

- `instance`: Your ServiceNow instance name (e.g., "dev12345" - without .service-now.com)
- `username`: ServiceNow username
- `password`: ServiceNow password
- `custom_parsers`: Dictionary mapping FileType enum values to BaseReader instances (REQUIRED)

**Optional (for OAuth2 password grant flow):**

- `client_id`: OAuth2 client ID (if provided, client_secret is also required)
- `client_secret`: OAuth2 client secret (if provided, client_id is also required)

If OAuth2 parameters are not provided, the reader will use basic authentication with username/password.

## Event System

The ServiceNow Knowledge Base reader uses LlamaIndex's standard instrumentation event system to provide detailed tracking of the loading process. Events are fired at various stages during knowledge base article retrieval and attachment processing.

### Available Events

- `SNOWKBTotalPagesEvent`: Fired when the total number of pages to process is determined
- `SNOWKBPageFetchStartEvent`: Fired when page data fetch starts
- `SNOWKBPageFetchCompletedEvent`: Fired when page data fetch completes successfully
- `SNOWKBPageSkippedEvent`: Fired when a page is skipped
- `SNOWKBPageFailedEvent`: Fired when page processing fails
- `SNOWKBAttachmentProcessingStartEvent`: Fired when attachment processing starts
- `SNOWKBAttachmentProcessedEvent`: Fired when attachment processing completes successfully
- `SNOWKBAttachmentSkippedEvent`: Fired when an attachment is skipped
- `SNOWKBAttachmentFailedEvent`: Fired when attachment processing fails

All events inherit from LlamaIndex's `BaseEvent` class and can be monitored using the standard LlamaIndex instrumentation dispatcher.

## Features

- Load knowledge base articles by sys_id or KB numbers
- Automatically download and process attachments
- **Requires custom parsers for different file types (HTML parser is mandatory)**
- LlamaIndex event-driven architecture for monitoring processing
- Configurable knowledge base table (defaults to `kb_knowledge`)
- Support for filtering by workflow state (defaults to "Published")
- Configurable temporary folder for file processing

## Required Custom Parsers

The reader requires custom parsers to process different file types. **At minimum, an HTML parser must be provided** for processing article bodies.

**Important**: The ServiceNow reader does not include built-in parsers. You must define your own custom parser classes that inherit from `BaseReader` and implement the `load_data` method.

### Example Custom Parser Implementation

```python
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from markitdown import MarkItDown
from typing import List, Union
import pathlib


class DocxParser(BaseReader):
    """DOCX parser using MarkItDown for text extraction."""

    def __init__(self):
        self.markitdown = MarkItDown()

    def load_data(
        self, file_path: Union[str, pathlib.Path], **kwargs
    ) -> List[Document]:
        """Load and parse a DOCX file."""
        result = self.markitdown.convert(source=file_path)

        return [
            Document(
                text=result.markdown, metadata={"file_path": str(file_path)}
            )
        ]


class HTMLParser(BaseReader):
    """Simple HTML parser for article body content."""

    def load_data(
        self, file_path: Union[str, pathlib.Path], **kwargs
    ) -> List[Document]:
        """Load and parse an HTML file."""
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Basic HTML to text conversion (you may want to use a more sophisticated parser)
        from html import unescape
        import re

        text = re.sub("<.*?>", "", html_content)
        text = unescape(text)

        return [Document(text=text, metadata={"file_path": str(file_path)})]
```

### Parser Requirements by File Type

**Required:**

- `FileType.HTML`: For processing article body content (MANDATORY)

**Recommended:**

- `FileType.PDF`: For PDF documents
- `FileType.DOCUMENT`: For Word documents (.docx)
- `FileType.TEXT`: For plain text files
- `FileType.SPREADSHEET`: For Excel files (.xlsx)
- `FileType.PRESENTATION`: For PowerPoint files (.pptx)

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
from llama_index.readers.service_now import SnowKBReader, FileType
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from typing import List, Union
import pathlib


# Define your custom HTML parser (REQUIRED)
class HTMLParser(BaseReader):
    """Simple HTML parser for article body content."""

    def load_data(
        self, file_path: Union[str, pathlib.Path], **kwargs
    ) -> List[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Basic HTML to text conversion
        from html import unescape
        import re

        text = re.sub("<.*?>", "", html_content)
        text = unescape(text)

        return [Document(text=text, metadata={"file_path": str(file_path)})]


# Custom parsers are REQUIRED - at minimum HTML parser must be provided
custom_parsers = {
    FileType.HTML: HTMLParser()  # Required for article body processing
}

reader = SnowKBReader(
    instance="dev12345",  # Instance name without .service-now.com
    custom_parsers=custom_parsers,  # REQUIRED parameter
    username="your_username",
    password="your_password",
    # Optional OAuth2 parameters:
    # client_id="your_client_id",
    # client_secret="your_client_secret",
)

# Load a specific article by sys_id
documents = reader.load_data(article_sys_id="your_article_sys_id")
```

```python
# Example that reads multiple KB articles by their numbers
from llama_index.readers.service_now import SnowKBReader, FileType
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from markitdown import MarkItDown
from typing import List, Union
import pathlib


# Define custom parsers (you must implement these yourself)
class HTMLParser(BaseReader):
    def load_data(
        self, file_path: Union[str, pathlib.Path], **kwargs
    ) -> List[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        from html import unescape
        import re

        text = re.sub("<.*?>", "", html_content)
        text = unescape(text)
        return [Document(text=text, metadata={"file_path": str(file_path)})]


class PDFParser(BaseReader):
    def load_data(
        self, file_path: Union[str, pathlib.Path], **kwargs
    ) -> List[Document]:
        # Implement your PDF parsing logic here
        # This is just a placeholder - use your preferred PDF library
        return [
            Document(
                text="PDF content here", metadata={"file_path": str(file_path)}
            )
        ]


# Custom parsers are REQUIRED - at minimum HTML parser must be provided
custom_parsers = {
    FileType.HTML: HTMLParser(),  # Required for article body processing
    FileType.PDF: PDFParser(),  # Optional: for PDF attachments
}

reader = SnowKBReader(
    instance="dev12345",  # Instance name without .service-now.com
    custom_parsers=custom_parsers,  # REQUIRED parameter
    username="your_username",
    password="your_password",
)

# Load articles by KB numbers
kb_numbers = ["KB0010001", "KB0010002", "KB0010003"]
documents = reader.load_data(numbers=kb_numbers)
```

```python
# Example with custom parsers and callbacks
from llama_index.readers.service_now import SnowKBReader, FileType
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from markitdown import MarkItDown
from typing import List, Union
import pathlib


# Define custom parsers for different file types (HTML is required)
class HTMLParser(BaseReader):
    def load_data(
        self, file_path: Union[str, pathlib.Path], **kwargs
    ) -> List[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        from html import unescape
        import re

        text = re.sub("<.*?>", "", html_content)
        text = unescape(text)
        return [Document(text=text, metadata={"file_path": str(file_path)})]


class PDFParser(BaseReader):
    def load_data(
        self, file_path: Union[str, pathlib.Path], **kwargs
    ) -> List[Document]:
        # Implement PDF parsing - example using PyMuPDF
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return [Document(text=text, metadata={"file_path": str(file_path)})]


class DocxParser(BaseReader):
    """DOCX parser using MarkItDown for text extraction."""

    def __init__(self):
        self.markitdown = MarkItDown()

    def load_data(
        self, file_path: Union[str, pathlib.Path], **kwargs
    ) -> List[Document]:
        result = self.markitdown.convert(source=file_path)
        return [
            Document(
                text=result.markdown, metadata={"file_path": str(file_path)}
            )
        ]


# Usage with ServiceNow Reader - Multiple file type parsers
custom_parsers = {
    FileType.HTML: HTMLParser(),  # Required for article body processing
    FileType.PDF: PDFParser(),  # For PDF attachments
    FileType.DOCUMENT: DocxParser(),  # For Word documents
}


# Define callback functions for processing
def process_attachment_callback(
    content_type: str, size_bytes: int, file_name: str
) -> tuple[bool, str]:
    # Return (should_process, reason)
    if size_bytes > 10 * 1024 * 1024:  # Skip files larger than 10MB
        return False, "File too large"
    return True, "Processing file"


def process_document_callback(kb_number: str) -> bool:
    # Return whether to process this document
    return kb_number.startswith(
        "KB001"
    )  # Only process KB numbers starting with KB001


reader = SnowKBReader(
    instance="dev12345",  # Instance name without .service-now.com
    custom_parsers=custom_parsers,  # REQUIRED parameter
    username="your_username",
    password="your_password",
    client_id="your_client_id",
    client_secret="your_client_secret",
    process_attachment_callback=process_attachment_callback,
    process_document_callback=process_document_callback,
    kb_table="kb_knowledge",  # Specify custom knowledge base table
    fail_on_error=False,  # Continue processing even if some articles fail
    custom_folder="/tmp/servicenow_parsing",  # Custom folder for temporary files
)

documents = reader.load_data(numbers=["KB0010001", "KB0010002"])
```

```python
# Example with LlamaIndex event monitoring
from llama_index.readers.service_now import SnowKBReader, FileType
from llama_index.readers.service_now.event import (
    SNOWKBPageFetchCompletedEvent,
    SNOWKBAttachmentProcessedEvent,
)
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.core.instrumentation import get_dispatcher
from typing import List, Union
import pathlib


# Define your custom parsers
class HTMLParser(BaseReader):
    def load_data(
        self, file_path: Union[str, pathlib.Path], **kwargs
    ) -> List[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        from html import unescape
        import re

        text = re.sub("<.*?>", "", html_content)
        text = unescape(text)
        return [Document(text=text, metadata={"file_path": str(file_path)})]


class PDFParser(BaseReader):
    def load_data(
        self, file_path: Union[str, pathlib.Path], **kwargs
    ) -> List[Document]:
        # Your PDF parsing implementation
        return [
            Document(
                text="PDF content", metadata={"file_path": str(file_path)}
            )
        ]


class MyEventHandler(BaseEventHandler):
    """Custom event handler for ServiceNow events."""

    def handle(self, event):
        """Handle incoming events."""
        if isinstance(event, SNOWKBPageFetchCompletedEvent):
            print(f"Processed page: {event.page_id}")
        elif isinstance(event, SNOWKBAttachmentProcessedEvent):
            print(f"Processed attachment: {event.attachment_name}")


# Set up event dispatcher
dispatcher = get_dispatcher()
handler = MyEventHandler()
dispatcher.add_event_handler(handler)

# Custom parsers are REQUIRED
custom_parsers = {
    FileType.HTML: HTMLParser(),  # Required for article body processing
    FileType.PDF: PDFParser(),  # Optional: for PDF attachments
}

reader = SnowKBReader(
    instance="dev12345",  # Instance name without .service-now.com
    custom_parsers=custom_parsers,  # REQUIRED parameter
    username="your_username",
    password="your_password",
    client_id="your_client_id",
    client_secret="your_client_secret",
)

documents = reader.load_data(article_sys_id="your_article_sys_id")
```

## Prerequisites

Before using this loader, make sure to install the required dependencies:

```bash
# Core dependency
pip install pysnc
```

**Note**: The specific dependencies depend on which parsing libraries you choose to use in your custom parser implementations. The ServiceNow reader itself only requires `pysnc`.

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
