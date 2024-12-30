# GCS File or Directory Loader

This loader parses any file stored on Google Cloud Storage (GCS), or the entire Bucket (with an optional prefix filter) if no particular file is specified. It now supports more advanced operations through the implementation of ResourcesReaderMixin and FileSystemReaderMixin.

## Features

- Parse single files or entire buckets from GCS
- List resources in GCS buckets
- Retrieve detailed information about GCS objects
- Load specific resources from GCS
- Read file content directly
- Supports various authentication methods
- Comprehensive logging for easier debugging
- Robust error handling for improved reliability

## Authentication

When initializing `GCSReader`, you may pass in your [GCP Service Account Key](https://cloud.google.com/iam/docs/keys-create-delete) in several ways:

1. As a file path (`service_account_key_path`)
2. As a JSON string (`service_account_key_json`)
3. As a dictionary (`service_account_key`)

If no credentials are provided, the loader will attempt to use default credentials.

## Usage

To use this loader, you need to pass in the name of your GCS Bucket. You can then either parse a single file by passing its key, or parse multiple files using a prefix.

```python
from llama_index import GCSReader
import logging

# Set up logging (optional, but recommended)
logging.basicConfig(level=logging.INFO)

# Initialize the reader
reader = GCSReader(
    bucket="scrabble-dictionary",
    key="dictionary.txt",  # Optional: specify a single file
    # prefix="subdirectory/",  # Optional: specify a prefix to filter files
    service_account_key_json="[SERVICE_ACCOUNT_KEY_JSON]",
)

# Load data
documents = reader.load_data()

# List resources in the bucket
resources = reader.list_resources()

# Get information about a specific resource
resource_info = reader.get_resource_info("dictionary.txt")

# Load a specific resource
specific_doc = reader.load_resource("dictionary.txt")

# Read file content directly
file_content = reader.read_file_content("dictionary.txt")

print(f"Loaded {len(documents)} documents")
print(f"Found {len(resources)} resources")
print(f"Resource info: {resource_info}")
print(f"Specific document: {specific_doc}")
print(f"File content length: {len(file_content)} bytes")
```

Note: If the file is nested in a subdirectory, the key should contain that, e.g., `subdirectory/input.txt`.

## Advanced Usage

All files are parsed with `SimpleDirectoryReader`. You may specify a custom `file_extractor`, relying on any of the loaders in the LlamaIndex library (or your own)!

```python
from llama_index import GCSReader, SimpleMongoReader

reader = GCSReader(
    bucket="my-bucket",
    file_extractor={
        ".mongo": SimpleMongoReader(),
        # Add more custom extractors as needed
    },
)
```

## Error Handling

The GCSReader now includes comprehensive error handling. You can catch exceptions to handle specific error cases:

```python
from google.auth.exceptions import DefaultCredentialsError

try:
    reader = GCSReader(bucket="your-bucket-name")
    documents = reader.load_data()
except DefaultCredentialsError:
    print("Authentication failed. Please check your credentials.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
```

## Logging

To get insights into the GCSReader's operations, configure logging in your application:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/). For more advanced usage, including custom file extractors, metadata extraction, and working with specific file types, please refer to the [LlamaIndex documentation](https://docs.llamaindex.ai/).
