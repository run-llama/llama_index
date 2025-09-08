# Microsoft SharePoint Reader

```bash
pip install llama-index-readers-microsoft-sharepoint
```

The loader loads files from a folder in a SharePoint site.

It also supports traversing recursively through sub-folders.

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

### Custom File Readers

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

### Page Reading Support

You can also load SharePoint pages (not just files) by setting `sharepoint_type="page"` and providing a `page_name` if you want to load a specific page.

```python
loader = SharePointReader(
    client_id="...",
    client_secret="...",
    tenant_id="...",
    sharepoint_type="page",
    page_name="<Page Name>",  # Optional: load a specific page
)

documents = loader.load_data(
    sharepoint_site_name="<Sharepoint Site Name>",
    # No need for sharepoint_folder_path when loading pages
)
```

---

## Notes

- The loader does not access other components of the SharePoint Site.
- If you use `custom_parsers`, you must also provide `custom_folder` (a directory for temporary files).
- For more advanced usage, see the docstrings in the code and the [examples](examples/) directory if available.
