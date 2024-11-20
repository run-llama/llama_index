# LlamaIndex Readers Integration: Document360

The `Document360Reader` class is a custom reader that interacts with the Document360 API to fetch articles. It processes these articles recursively and allows further handling via custom callback functions, while also handling rate limiting and errors.

## Installation

```bash
pip install llama-index-readers-document360
```

## Usage

```py
from document360_reader import Document360Reader

reader = Document360Reader(api_key="your_api_key")

# Load data
documents = reader.load_data()

# Use the documents as needed
for doc in documents:
    print(doc.text)
```

## Class Initialization

```py
def __init__(
    self,
    api_key: str,
    should_process_project_version=None,
    should_process_category=None,
    should_process_article=None,
    handle_batch_finished=None,
    handle_rate_limit_error=None,
    handle_request_http_error=None,
    handle_category_processing_started=None,
    handle_article_processing_started=None,
    handle_article_processing_error=None,
    handle_load_data_error=None,
    article_to_custom_document=None,
    rate_limit_num_retries=10,
    rate_limit_retry_wait_time=30,
):
    pass
```

`api_key`: Your Document360 API key (required).
`should_process_project_version`: Callback to determine whether to process a project version.
`should_process_category`: Callback to determine whether to process a category.
`should_process_article`: Callback to determine whether to process an article.
`handle_batch_finished`: Callback executed after all articles are processed.
`handle_rate_limit_error`: Callback for handling rate limit errors.
`handle_request_http_error`: Callback for handling HTTP errors.
`handle_category_processing_started`: Callback triggered when category processing starts.
`handle_article_processing_started`: Callback triggered when article processing starts.
`handle_article_processing_error`: Callback for handling errors during article processing.
`handle_load_data_error`: Callback for handling errors during data loading.
`article_to_custom_document`: Custom transformation function to map an article to a document.
`rate_limit_num_retries`: Number of retry attempts when hitting rate limits.
`rate_limit_retry_wait_time`: Time to wait (in seconds) between retries after a rate limit error.

## Referencing entities

```py
from llama_index.readers.document360.entities import (
    Article,
    ArticleSlim,
    Category,
    ProjectVersion,
)


def handle_category_processing_started(category: Category):
    logging.info(f"Started processing category: {category}")


def handle_article_processing_started(article: Article):
    logging.info(f"Processing article: {article}")
```

All the fields in the entities are marked as Optional. This is because the actual API responses from Document360 sometimes do not match the expected schema mentioned in the API documentation.

## Referencing errors

```py
from llama_index.readers.document360.errors import (
    RetryError,
    HTTPError,
    RateLimitException,
)

reader = Document360Reader(api_key="your_api_key")

try:
    reader.load_data()
except RetryError as e:
    logging.info(f"Retry Error: {e}")
```
