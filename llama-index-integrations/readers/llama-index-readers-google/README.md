# LlamaIndex Integration: Google Readers

Effortlessly incorporate Google-based data loaders into your Python workflow using LlamaIndex. Unlock the potential of various readers to enhance your data loading capabilities. Below are examples of integrating Google Docs and Google Sheets readers:

### Google Docs Reader

```python
from llama_index.readers.google import GoogleDocsReader

# Specify the document IDs you want to load
document_ids = ["<document_id>"]

# Load data from Google Docs
documents = GoogleDocsReader().load_data(document_ids=document_ids)
```

### Google Sheets Reader

```python
from llama_index.readers.google import GoogleSheetsReader

# Specify the list of sheet IDs you want to load
list_of_sheets = ["spreadsheet_id"]

# Create a Google Sheets Reader instance
sheets_reader = GoogleSheetsReader()

# Load data into Pandas DataFrames
dataframes = sheets_reader.load_data_in_pandas(list_of_sheets)
```

Integrate these readers seamlessly to efficiently manage and process your data within your Python environment, providing a robust foundation for your data-driven workflows with LlamaIndex.
