# Microsoft SharePoint Reader

```bash
pip install llama-index-readers-microsoft-sharepoint
```

The loader loads files from a folder in a SharePoint site.

It also supports traversing recursively through sub-folders.

## ✨ New Features

- **📄 SharePoint Page Reading**: Load SharePoint site pages as documents
- **🔧 Custom File Parsers**: Use specialized parsers for different file types (PDF, DOCX, HTML, etc.)
- **📊 Event System**: Monitor document processing with real-time events
- **🎯 Document Callbacks**: Filter and process documents with custom logic
- **⚙️ Error Handling**: Configurable error handling behavior
- **🚀 Enhanced Performance**: Optimized loading with parallel processing support

---

## Prerequisites

### App Authentication using Microsoft Entra ID (formerly Azure AD)

1. You need to create an App Registration in Microsoft Entra ID. Refer [here](https://learn.microsoft.com/en-us/azure/healthcare-apis/register-application)
2. API Permissions for the created app:
   - Microsoft Graph → Application Permissions → **Sites.Read.All** (**Grant Admin Consent**)  
     *(Allows access to all sites in the tenant)*
   - **OR**  
     Microsoft Graph → Application Permissions → **Sites.Selected** (**Grant Admin Consent**)  
     *(Allows access only to specific sites you select and grant permissions for)*
   - Microsoft Graph → Application Permissions → Files.Read.All (**Grant Admin Consent**)
   - Microsoft Graph → Application Permissions → BrowserSiteLists.Read.All (**Grant Admin Consent**)

> **Note:**  
> If you use `Sites.Selected`, you must grant your app access to the specific SharePoint site(s) via the SharePoint admin center.  
> See [Grant access to a specific site](https://learn.microsoft.com/en-us/sharepoint/dev/solution-guidance/security-apponly-azuread#grant-access-to-a-specific-site) for details.

More info on Microsoft Graph APIs - [Refer here](https://learn.microsoft.com/en-us/graph/permissions-reference)

---

## Usage

To use this loader, you need the `client_id`, `client_secret`, and `tenant_id` of the registered app in Microsoft Azure Portal.

This loader loads the files present in a specific folder in SharePoint.

If the files are present in the `Test` folder in a SharePoint Site under the `root` directory, then the input for the loader for `sharepoint_folder_path` is `Test`.

![FilePath](file_path_info.png)

### Example: Using `sharepoint_site_name`

```python
from llama_index.readers.microsoft_sharepoint import SharePointReader

loader = SharePointReader(
    client_id="<Client ID of the app>",
    client_secret="<Client Secret of the app>",
    tenant_id="<Tenant ID of the Microsoft Azure Directory>",
)

documents = loader.load_data(
    sharepoint_site_name="<Sharepoint Site Name>",
    sharepoint_folder_path="<Folder Path>",
    recursive=True,
)
```

### Example: Using `sharepoint_host_name` and `sharepoint_relative_url`

If you have only been granted access to a specific site (using `Sites.Selected`), you can use the site host name and relative URL:

```python
loader = SharePointReader(
    client_id="<Client ID of the app>",
    client_secret="<Client Secret of the app>",
    tenant_id="<Tenant ID of the Microsoft Azure Directory>",
    sharepoint_host_name="contoso.sharepoint.com",
    sharepoint_relative_url="sites/YourSiteName",
)

documents = loader.load_data(
    sharepoint_folder_path="<Folder Path>",
    recursive=True,
)
```

---

## Advanced Features

### 🔧 Custom File Parsers

You can use custom file readers for specific file types (e.g., PDF, DOCX, HTML, etc.) by passing the `custom_parsers` argument. This allows you to control how different file types are parsed.

```python
from llama_index.readers.microsoft_sharepoint.file_parsers import (
    PDFReader, HTMLReader, DocxReader, PptxReader, CSVReader, ExcelReader, ImageReader
)
from llama_index.readers.microsoft_sharepoint.event import FileType

custom_parsers = {
    FileType.PDF: PDFReader(),
    FileType.HTML: HTMLReader(),
    FileType.DOCUMENT: DocxReader(),
    FileType.PRESENTATION: PptxReader(),
    FileType.CSV: CSVReader(),
    FileType.SPREADSHEET: ExcelReader(),
    FileType.IMAGE: ImageReader(),
}

loader = SharePointReader(
    client_id="...",
    client_secret="...",
    tenant_id="...",
    custom_parsers=custom_parsers,
    custom_folder="/tmp",  # Directory for temporary files
)
```

### 📄 SharePoint Page Reading

You can load SharePoint pages (not just files) by setting `sharepoint_type="page"` and providing a `page_name` if you want to load a specific page.

```python
from llama_index.readers.microsoft_sharepoint.base import SharePointType

# Load all pages from a site
loader = SharePointReader(
    client_id="...",
    client_secret="...",
    tenant_id="...",
    sharepoint_type=SharePointType.PAGE,
)

documents = loader.load_data(
    sharepoint_site_name="<Sharepoint Site Name>",
    download_dir="/tmp/pages"  # Required for page content processing
)

# Load a specific page
loader = SharePointReader(
    client_id="...",
    client_secret="...",
    tenant_id="...",
    sharepoint_type=SharePointType.PAGE,
    page_name="<Page Name>",
)
```

### 🎯 Document Filtering with Callbacks

Use callbacks to filter or modify documents during processing:

```python
def should_process_document(file_name: str) -> bool:
    """Filter out certain files based on name patterns."""
    return not file_name.startswith('temp_') and not file_name.endswith('.tmp')

loader = SharePointReader(
    client_id="...",
    client_secret="...",
    tenant_id="...",
    process_document_callback=should_process_document,
)
```

### 📊 Event System for Monitoring

Monitor document processing with real-time events:

```python
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.readers.microsoft_sharepoint.event import (
    PageDataFetchStartedEvent,
    PageDataFetchCompletedEvent,
    PageSkippedEvent,
    PageFailedEvent,
)

class SharePointEventHandler(BaseEventHandler):
    def handle(self, event):
        if isinstance(event, PageDataFetchStartedEvent):
            print(f"Started processing: {event.page_id}")
        elif isinstance(event, PageDataFetchCompletedEvent):
            print(f"Completed processing: {event.page_id}")
        elif isinstance(event, PageSkippedEvent):
            print(f"Skipped: {event.page_id}")
        elif isinstance(event, PageFailedEvent):
            print(f"Failed: {event.page_id} - {event.error}")

# Register event handler
dispatcher = get_dispatcher("llama_index.readers.microsoft_sharepoint.base")
dispatcher.add_event_handler(SharePointEventHandler())

# Now load data with event monitoring
documents = loader.load_data(sharepoint_site_name="YourSite")
```

### ⚙️ Error Handling

Configure how the reader handles errors:

```python
# Fail immediately on any error (default)
loader = SharePointReader(
    client_id="...",
    client_secret="...",
    tenant_id="...",
    fail_on_error=True,
)

# Continue processing even if some files fail
loader = SharePointReader(
    client_id="...",
    client_secret="...",
    tenant_id="...",
    fail_on_error=False,  # Skip failed files and continue
)
```

---

## 📋 Installation Options

### Basic Installation
```bash
pip install llama-index-readers-microsoft-sharepoint
```

### With File Parser Support
For enhanced file parsing capabilities (PDF, DOCX, images, etc.):
```bash
pip install "llama-index-readers-microsoft-sharepoint[file_parsers]"
```

This includes additional dependencies:
- `pytesseract` - For OCR in images
- `pdf2image` - For PDF processing  
- `python-pptx` - For PowerPoint files
- `docx2txt` - For Word documents
- `pandas` - For Excel/CSV files
- `beautifulsoup4` - For HTML parsing
- `Pillow` - For image processing

---

## 🔧 Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `sharepoint_type` | `SharePointType` | Type of SharePoint content (`DRIVE` or `PAGE`) | `DRIVE` |
| `custom_parsers` | `Dict[FileType, Any]` | Custom parsers for specific file types | `{}` |
| `custom_folder` | `str` | Directory for temporary files (required with custom_parsers) | `None` |
| `process_document_callback` | `Callable` | Function to filter/process documents | `None` |
| `fail_on_error` | `bool` | Whether to stop on first error or continue | `True` |

---

## Notes

- The loader does not access other components of the SharePoint Site.
- If you use `custom_parsers`, you must also provide `custom_folder` (a directory for temporary files).
- SharePoint page reading requires a download directory for content processing.
- Event monitoring is optional but provides valuable insights into processing status.
- For more advanced usage, see the docstrings in the code and the test files for examples.
