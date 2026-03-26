# Microsoft SharePoint Reader

```bash
pip install llama-index-readers-microsoft-sharepoint
```

The loader loads the files from a folder in SharePoint site or SharePoint Site Pages.

It also supports traversing recursively through the sub-folders.

## Prerequisites

### App Authentication using Microsoft Entra ID (formerly Azure AD)

1. You need to create an App Registration in Microsoft Entra ID. Refer [here](https://learn.microsoft.com/en-us/azure/healthcare-apis/register-application)
2. API Permissions for the created app:
   - Microsoft Graph → Application Permissions → **Sites.Read.All** (**Grant Admin Consent**)
     _(Allows access to all sites in the tenant)_
   - **OR**
     Microsoft Graph → Application Permissions → **Sites.Selected** (**Grant Admin Consent**)
     _(Allows access only to specific sites you select and grant permissions for)_
   - Microsoft Graph → Application Permissions → Files.Read.All (**Grant Admin Consent**)
   - Microsoft Graph → Application Permissions → BrowserSiteLists.Read.All (**Grant Admin Consent**)

> **Note:**
> If you use `Sites.Selected`, you must grant your app access to the specific SharePoint site(s) via the SharePoint admin center.
> See [Grant access to a specific site](https://learn.microsoft.com/en-us/sharepoint/dev/solution-guidance/security-apponly-azuread#grant-access-to-a-specific-site) for details.

More info on Microsoft Graph APIs - [Refer here](https://learn.microsoft.com/en-us/graph/permissions-reference)

## Usage

To use this loader `client_id`, `client_secret` and `tenant_id` of the registered app in Microsoft Azure Portal is required.

### Loading Files from SharePoint Drive

This loader loads the files present in a specific folder in SharePoint.

If the files are present in the `Test` folder in SharePoint Site under `root` directory, then the input for the loader for `file_path` is `Test`

![FilePath](file_path_info.png)

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

### Using Sites.Selected Permission

If you have only been granted access to a specific site (using `Sites.Selected`), you can use the site host name and relative URL instead of the site name:

```python
from llama_index.readers.microsoft_sharepoint import SharePointReader

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

### Loading SharePoint Site Pages

You can also load SharePoint Site Pages as documents by setting `sharepoint_type` to `PAGE`:

```python
from llama_index.readers.microsoft_sharepoint import (
    SharePointReader,
    SharePointType,
)

loader = SharePointReader(
    client_id="<Client ID of the app>",
    client_secret="<Client Secret of the app>",
    tenant_id="<Tenant ID of the Microsoft Azure Directory>",
    sharepoint_site_name="<Sharepoint Site Name>",
    sharepoint_host_name="<your-tenant>.sharepoint.com",
    sharepoint_relative_url="/sites/<YourSite>",
    sharepoint_type=SharePointType.PAGE,
)

# Load all pages
documents = loader.load_data()

# Or load a specific page by ID
loader.sharepoint_file_id = "<page_id>"
documents = loader.load_data()
```

### Filtering Pages with Callbacks

You can filter which pages to process using the `process_document_callback`:

```python
def page_filter(page_name: str) -> bool:
    # Only process pages that don't start with "Draft"
    return not page_name.startswith("Draft")


loader = SharePointReader(
    client_id="<Client ID>",
    client_secret="<Client Secret>",
    tenant_id="<Tenant ID>",
    sharepoint_site_name="<Site Name>",
    sharepoint_type=SharePointType.PAGE,
    process_document_callback=page_filter,
)
```

### Error Handling

Control error behavior with `fail_on_error`:

```python
loader = SharePointReader(
    client_id="<Client ID>",
    client_secret="<Client Secret>",
    tenant_id="<Tenant ID>",
    fail_on_error=False,  # Log errors and continue instead of raising
)
```

## Instrumentation Events

The SharePoint reader emits events during page processing for monitoring:

```python
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.readers.microsoft_sharepoint import (
    TotalPagesToProcessEvent,
    PageDataFetchCompletedEvent,
    PageFailedEvent,
)


class SharePointEventHandler(BaseEventHandler):
    def handle(self, event):
        if isinstance(event, TotalPagesToProcessEvent):
            print(f"Processing {event.total_pages} pages...")
        elif isinstance(event, PageDataFetchCompletedEvent):
            print(f"Completed: {event.page_id}")
        elif isinstance(event, PageFailedEvent):
            print(f"Failed: {event.page_id} - {event.error}")


dispatcher = get_dispatcher("llama_index.readers.microsoft_sharepoint.base")
dispatcher.add_event_handler(SharePointEventHandler())
```

Available events:

- `TotalPagesToProcessEvent`: Total number of pages to process
- `PageDataFetchStartedEvent`: Page processing started
- `PageDataFetchCompletedEvent`: Page successfully processed
- `PageSkippedEvent`: Page skipped (via callback)
- `PageFailedEvent`: Page processing failed
