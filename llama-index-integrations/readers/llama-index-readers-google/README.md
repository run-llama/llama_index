# LlamaIndex Integration: Google Readers

Effortlessly incorporate Google-based data loaders into your Python workflow using LlamaIndex. It now supports more advanced operations through the implementation of ResourcesReaderMixin and FileSystemReaderMixin.
Unlock the potential of various readers to enhance your data loading capabilities, including:

- Google Calendar
- Google Chat
- Google Docs
- Google Drive
- Gmail
- Google Keep
- Google Maps
- Google Sheets

## Installation

```bash
pip install llama-index-readers-google
```

## Authentication

You will need a `credentials.json` file from Google Cloud to interact with Google Services. To get this file, follow these steps:

- Create a new project in the [Google Cloud Console](https://console.cloud.google.com/)
- Go to APIs & Services -> Library and search for the API you want, e.g. Gmail
- Go to APIs & Services -> Credentials and create a new OAuth client ID
  - Application type: Web application
  - Authorized redirect URIs: http://localhost:8080/ (the last slash seems important)
- Go to APIs & Services -> OAuth consent screen and make the app external, which allows you to connect your personal Google data once you explicitly add yourself as an allowed test user
- Download the credentials JSON file from this screen and save it as `credentials.json` in the root of your project

See [this example](https://github.com/run-llama/gmail-extractor/blob/main/gmail.py) for a sample of code that successfully authenticates with Gmail once you have the `credentials.json` file.

## Examples

### Google Drive Reader

```python
from llama_index.readers.google import GoogleDriveReader

# Initialize the reader
reader = GoogleDriveReader(
    folder_id="folder_id",
    service_account_key="[SERVICE_ACCOUNT_KEY_JSON]",
)

# Load data
documents = reader.load_data()

# List resources in the drive
resources = reader.list_resources()

# Get information about a specific resource
resource_info = reader.get_resource_info("file.txt")

# Load a specific resource
specific_doc = reader.load_resource("file.txt")

# Read file content directly
file_content = reader.read_file_content("path/to/file.txt")

print(f"Loaded {len(documents)} documents")
print(f"Found {len(resources)} resources")
print(f"Resource info: {resource_info}")
print(f"Specific document: {specific_doc}")
print(f"File content length: {len(file_content)} bytes")
```

### Google Docs Reader

```python
from llama_index.readers.google import GoogleDocsReader

# Specify the document IDs you want to load
document_ids = ["<document_id>"]

# Load data from Google Docs
documents = GoogleDocsReader().load_data(document_ids=document_ids)
```

### Google Sheets Reader (Documents and Dataframes)

```python
from llama_index.readers.google import GoogleSheetsReader

# Specify the list of sheet IDs you want to load
list_of_sheets = ["spreadsheet_id"]

# Create a Google Sheets Reader instance
sheets_reader = GoogleSheetsReader()

# Load data into Pandas in Data Classes of choice (Documents or Dataframes)
documents = sheets.load_data(list_of_sheets)
dataframes = sheets_reader.load_data_in_pandas(list_of_sheets)
```

Integrate these readers seamlessly to efficiently manage and process your data within your Python environment, providing a robust foundation for your data-driven workflows with LlamaIndex.

### Google Maps Text Search Reader

```python
from llama_index.readers.google import GoogleMapsTextSearchReader
from llama_index.core import VectorStoreIndex

loader = GoogleMapsTextSearchReader()
documents = loader.load_data(
    text="I want to eat quality Turkish food in Istanbul",
    number_of_results=160,
)


index = VectorStoreIndex.from_documents(documents)
index.query("Which Turkish restaurant has the best reviews?")
```

### Google Chat Reader

```py
from llama_index.readers.google import GoogleChatReader
from llama_index.core import VectorStoreIndex

space_names = ["<CHAT_ID>"]
chatReader = GoogleChatReader()
docs = chatReader.load_data(space_names=space_names)
index = VectorStoreIndex.from_documents(docs)
query_eng = index.as_query_engine()
print(query_eng.query("What was this conversation about?"))
```
