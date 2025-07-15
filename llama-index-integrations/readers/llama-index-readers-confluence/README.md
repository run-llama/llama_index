# Confluence Loader

```bash
pip install llama-index-readers-confluence
```

This loader loads pages from a given Confluence cloud instance. The user needs to specify the base URL for a Confluence
instance to initialize the ConfluenceReader - base URL needs to end with `/wiki`.

The user can optionally specify OAuth 2.0 credentials to authenticate with the Confluence instance. If no credentials are
specified, the loader will look for `CONFLUENCE_API_TOKEN` or `CONFLUENCE_USERNAME`/`CONFLUENCE_PASSWORD` environment variables
to proceed with basic authentication.

> [!NOTE]
> Keep in mind `CONFLUENCE_PASSWORD` is not your actual password, but an API Token obtained here: https://id.atlassian.com/manage-profile/security/api-tokens.

The following order is used for checking authentication credentials:

1. `oauth2`
2. `api_token`
3. `cookies`
4. `user_name` and `password`
5. Environment variable `CONFLUENCE_API_TOKEN`
6. Environment variable `CONFLUENCE_USERNAME` and `CONFLUENCE_PASSWORD`

For more on authenticating using OAuth 2.0, checkout:

- https://atlassian-python-api.readthedocs.io/index.html
- https://developer.atlassian.com/cloud/confluence/oauth-2-3lo-apps/

Confluence pages are obtained through one of 4 four mutually exclusive ways:

1. `page_ids`: Load all pages from a list of page ids
2. `space_key`: Load all pages from a space
3. `label`: Load all pages with a given label
4. `cql`: Load all pages that match a given CQL query (Confluence Query Language https://developer.atlassian.com/cloud/confluence/advanced-searching-using-cql/ ).

When `page_ids` is specified, `include_children` will cause the loader to also load all descendent pages.
When `space_key` is specified, `page_status` further specifies the status of pages to load: None, 'current', 'archived', 'draft'.

limit (int): Deprecated, use `max_num_results` instead.

max_num_results (int): Maximum number of results to return. If None, return all results. Requests are made in batches to achieve the desired number of results.

start(int): Which offset we should jump to when getting pages, only works with space_key

cursor(str): An alternative to start for cql queries, the cursor is a pointer to the next "page" when searching atlassian products. The current one after a search can be found with `get_next_cursor()`

User can also specify a boolean `include_attachments` to
include attachments, this is set to `False` by default, if set to `True` all attachments will be downloaded and
ConfluenceReader will extract the text from the attachments and add it to the Document object.
Currently supported attachment types are: PDF, PNG, JPEG/JPG, SVG, Word and Excel.

## Advanced Configuration

The ConfluenceReader supports several advanced configuration options for customizing the reading behavior:

**Custom Parsers**: You can provide custom parsers for specific file types using the `custom_parsers` parameter. This allows you to override the default parsing behavior for attachments of different types.

**Processing Callbacks**:

- `process_attachment_callback`: A callback function to control which attachments should be processed. The function receives the media type and file size as parameters and should return a tuple of `(should_process: bool, reason: str)`.
- `process_document_callback`: A callback function to control which documents should be processed. The function receives the page ID as a parameter and should return a boolean indicating whether to process the document.

**File Management**:

- `custom_folder`: Specify a custom directory for storing temporary files during processing. Can only be used when `custom_parsers` are provided. Defaults to the current working directory if `custom_parsers` are used.

**Error Handling**:

- `fail_on_error` (default: True): Whether to raise exceptions when encountering processing errors or continue with warnings and skip problematic content.

**Logging**:

- `logger`: Provide a custom logger instance for controlling log output during the reading process.

**Event Monitoring (Observer Pattern)**:
The ConfluenceReader includes an observer system that emits events during document and attachment processing. This allows you to monitor progress, handle errors, or integrate with external systems like databases or message queues.

Available event types:

- `TOTAL_PAGES_TO_PROCESS`: When the total number of pages to process is determined
- `PAGE_DATA_FETCH_STARTED`: When processing of a page begins
- `PAGE_DATA_FETCH_COMPLETED`: When a page is successfully processed
- `PAGE_FAILED`: When page processing fails
- `PAGE_SKIPPED`: When a page is skipped due to callback decision
- `ATTACHMENT_PROCESSING_STARTED`: When attachment processing begins
- `ATTACHMENT_PROCESSED`: When an attachment is successfully processed
- `ATTACHMENT_SKIPPED`: When an attachment is skipped
- `ATTACHMENT_FAILED`: When attachment processing fails

You can subscribe to specific events using `reader.observer.subscribe(event_name, callback)` or to all events using `reader.observer.subscribe_all(callback)`.

Hint: `space_key` and `page_id` can both be found in the URL of a page in Confluence - https://yoursite.atlassian.com/wiki/spaces/<space_key>/pages/<page_id>

## Usage

Here's an example usage of the ConfluenceReader.

```python
# Example that reads the pages with the `page_ids`
from llama_index.readers.confluence import ConfluenceReader

token = {"access_token": "<access_token>", "token_type": "<token_type>"}
oauth2_dict = {"client_id": "<client_id>", "token": token}

base_url = "https://yoursite.atlassian.com/wiki"

page_ids = ["<page_id_1>", "<page_id_2>", "<page_id_3"]
space_key = "<space_key>"

reader = ConfluenceReader(
    base_url=base_url,
    oauth2=oauth2_dict,
    client_args={"backoff_and_retry": True},
)
documents = reader.load_data(
    space_key=space_key, include_attachments=True, page_status="current"
)
documents.extend(
    reader.load_data(
        page_ids=page_ids, include_children=True, include_attachments=True
    )
)
```

```python
# Example that fetches the first 5, then the next 5 pages from a space
from llama_index.readers.confluence import ConfluenceReader

token = {"access_token": "<access_token>", "token_type": "<token_type>"}
oauth2_dict = {"client_id": "<client_id>", "token": token}

base_url = "https://yoursite.atlassian.com/wiki"

space_key = "<space_key>"

reader = ConfluenceReader(base_url=base_url, oauth2=oauth2_dict)
documents = reader.load_data(
    space_key=space_key,
    include_attachments=True,
    page_status="current",
    start=0,
    max_num_results=5,
)
documents.extend(
    reader.load_data(
        space_key=space_key,
        include_children=True,
        include_attachments=True,
        start=5,
        max_num_results=5,
    )
)
```

```python
# Example that fetches the first 5 results from a cql query, the uses the cursor to pick up on the next element
from llama_index.readers.confluence import ConfluenceReader

token = {"access_token": "<access_token>", "token_type": "<token_type>"}
oauth2_dict = {"client_id": "<client_id>", "token": token}

base_url = "https://yoursite.atlassian.com/wiki"

cql = f'type="page" AND label="devops"'

reader = ConfluenceReader(base_url=base_url, oauth2=oauth2_dict)
documents = reader.load_data(cql=cql, max_num_results=5)
cursor = reader.get_next_cursor()
documents.extend(reader.load_data(cql=cql, cursor=cursor, max_num_results=5))
```

```python
# Example with advanced configuration options
from llama_index.readers.confluence import ConfluenceReader
import logging


# Custom callback to filter attachments by size and type
def attachment_filter(
    media_type: str, file_size: int, title: str
) -> tuple[bool, str]:
    # Skip large files (>10MB)
    if file_size > 10 * 1024 * 1024:
        return False, f"File too large: {file_size} bytes"

    # Skip certain file types
    if media_type in ["application/x-zip-compressed", "application/zip"]:
        return False, f"Unsupported file type: {media_type}"

    return True, ""


# Custom callback to filter documents
def document_filter(page_id: str) -> bool:
    # Skip specific pages by ID
    excluded_pages = ["123456", "789012"]
    return page_id not in excluded_pages


# Setup custom logger
logger = logging.getLogger("confluence_reader")
logger.setLevel(logging.INFO)

reader = ConfluenceReader(
    base_url="https://yoursite.atlassian.com/wiki",
    api_token="your_api_token",
    process_attachment_callback=attachment_filter,
    process_document_callback=document_filter,
    custom_folder="/tmp/confluence_files",  # Custom temp directory
    logger=logger,
    fail_on_error=False,  # Continue processing even if some content fails
)

documents = reader.load_data(space_key="MYSPACE", include_attachments=True)
```

```python
# Example using the Observer pattern for event monitoring
from llama_index.readers.confluence import ConfluenceReader
from llama_index.readers.confluence.event import EventName


# Event handler functions
def handle_page_started(event):
    print(f"Started processing page: {event.page_id}")


def handle_page_completed(event):
    print(f"Completed processing page: {event.page_id}")
    print(f"Document title: {event.document.metadata.get('title', 'Unknown')}")


def handle_attachment_processed(event):
    print(f"Processed attachment: {event.attachment_name}")
    print(f"Attachment type: {event.attachment_type}")
    print(f"Attachment size: {event.attachment_size}")


def handle_processing_failed(event):
    print(f"Processing failed: {event.error}")


def handle_all_events(event):
    """General event handler that logs all events"""
    print(f"Event: {event.name}, Page ID: {event.page_id}")


# Create reader and set up event monitoring
reader = ConfluenceReader(
    base_url="https://yoursite.atlassian.com/wiki", api_token="your_api_token"
)

# Subscribe to specific events
reader.observer.subscribe(
    EventName.PAGE_DATA_FETCH_STARTED, handle_page_started
)
reader.observer.subscribe(
    EventName.PAGE_DATA_FETCH_COMPLETED, handle_page_completed
)
reader.observer.subscribe(
    EventName.ATTACHMENT_PROCESSED, handle_attachment_processed
)
reader.observer.subscribe(EventName.PAGE_FAILED, handle_processing_failed)
reader.observer.subscribe(
    EventName.ATTACHMENT_FAILED, handle_processing_failed
)

# Or subscribe to all events with a single handler
# reader.observer.subscribe_all(handle_all_events)

# Load data - events will be emitted during processing
documents = reader.load_data(space_key="MYSPACE", include_attachments=True)

print(f"Processing completed. Total documents: {len(documents)}")
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
