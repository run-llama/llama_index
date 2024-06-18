# Microsoft OneDrive Loader

```bash
pip install llama-index-readers-microsoft-onedrive
```

This loader reads files from:

- Microsoft OneDrive Personal [(https://onedrive.live.com/)](https://onedrive.live.com/) and
- Microsoft OneDrive for Business [(https://portal.office.com/onedrive)](https://portal.office.com/onedrive).

It supports recursively traversing and downloading files from subfolders and provides capability to download only files with specific mime types. To use this loader, you need to pass in a list of file/folder id or file/folder paths.

#### Subfolder traversing (enabled by default)

To disable: `loader.load_data(recursive = False)`

#### Mime types

You can also filter the files by the mimeType e.g.: `mime_types=["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]`

### Authenticaton

OneDriveReader supports following two **MSAL authentication**:

#### 1. User Authentication: Browser based authentication:

- You need to create a app registration in Microsoft Entra (formerly Azure Active Directory)
- For interactive authentication to work, a browser is used to authenticate, hence the registered application should have a **redirect URI** set to _'https://localhost'_ under mobile and native applications.
- This mode of authentication is not suitable for CI/CD or other background service scenarios where manual authentication isn't feasible.
- API Permission required for registered app:
  > Microsoft Graph --> Delegated Permission -- > Files.Read.All

#### 2. App Authentication: Client ID & Client Secret based authentication

- You need to create a app registration in Microsoft Entra (formerly Azure Active Directory)
- For silent authentication to work, You need to create a client secret as well for the app.
- This mode of authnetication is not supported by Microsoft currently for OneDrive Personal, hence this can be used only for OneDrive for Business(Microsoft 365).
- API Permission required for registered app:

  > Microsoft Graph --> Application Permissions -- > Files.Read.All (**Grant Admin Consent**)

  > Microsoft Graph --> Application Permissions -- > User.Read.All (**Grant Admin Consent**)

## Usage

### OneDrive Personal

https://onedrive.live.com/

> Note: If you trying to connect to OneDrive Personal you can initialize OneDriveReader with just your client*id and interactive login. Microsoft \_doesn't* support App authentication for OneDrive Personal currently.

#### folder_id

You can extract a folder_id directly from its URL.

For example, the folder_id of `https://onedrive.live.com/?id=B5AF52B769DFDE4%216107&cid=0B5AF52B769DFDdRE4` is `B5AF52B769DFDE4%216107`.

#### file_id

You can extract a file_id directly from its preview URL.

For example, the file_id of `https://onedrive.live.com/?cid=0B5AF52BE769DFDE4&id=B5AF52B769DFDE4%216106&parId=root&o=OneUp` is `B5AF52B769DFDE4%216106`.

#### OneDrive Personal Example Usage:

```python
from llama_index.readers.microsoft_onedrive import OneDriveReader

# User Authentication flow: Replace client id with your own id
loader = OneDriveReader(client_id="82ee706e-2439-47fa-877a-95048ead9318")

# APP Authentication flow: NOT SUPPORTED By Microsoft

#### Get all documents including subfolders.
documents = loader.load_data()

#### Get documents using folder_id , to exclude traversing subfolders explicitly set the recursive flag to False, default is True
documents = loader.load_data(folder_id="folderid", recursive=False)

#### Using file ids
documents = loader.load_data(file_ids=["fileid1", "fileid2"])
```

### OneDrive For Business

https://portal.office.com/onedrive

> Note: If you are an organization trying to connect to OneDrive for Business (Part of Microsoft 365), you need to:

1. Initialize OneDriveReader with correct **tenant_id**, along with a client_id and client_Secret registered for the tenant.
2. Invoke the load_data method with **userprincipalname** (org provided email in most cases)

#### folder_path

The relative pathof subfolder from the root folder(Documents).

For example:

- The path of 1st level subfolder with name "drice co" (within root folder) with URL of `https://foobar-my.sharepoint.com/personal/godwin_foobar_onmicrosoft_com/_layouts/15/onedrive.aspx?id=/personal/godwin_foobar_onmicrosoft_com/Documents/drice%20co/test` is **drice%20co**

- The path of 2nd level subfolder "test" (within drice co subfolder) with URL of `https://foobar-my.sharepoint.com/personal/godwin_foobar_onmicrosoft_com/_layouts/15/onedrive.aspx?id=/personal/godwin_foobar_onmicrosoft_com/Documents/drice%20co/test` is **drice%20co/test**

#### file_path

The relatve path of files from the root folder(Documents).

For example, the path of file "demo_doc.docx" within test subfolder from previous example with url of `https://foobar-my.sharepoint.com/personal/godwin_foobar_onmicrosoft_com/_layouts/15/onedrive.aspx?id=/personal/godwin_foobar_onmicrosoft_com/Documents/drice%20co/test/demo_doc.docx` is **drice%20co/test/demo_doc.docx**

#### OneDrive For Business Example Usage:

```python
from llama_index.readers.microsoft_onedrive import OneDriveReader

loader = OneDriveReader(
    client_id="82ee706e-2439-47fa-877a-95048ead9318",
    tenant_id="02ee706f-2439-47fa-877a-95048ead9318",
    client_secret="YOUR_SECRET",
)

#### Get all docx or pdf documents (subfolders included).
documents = loader.load_data(
    mime_types=[
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/pdf",
    ],
    userprincipalname="godwin@foobar.onmicrosoft.com",
)

#### Get all documents from a folder of mentioned user's onedrive for business
documents = loader.load_data(
    folder_path="subfolder/subfolder2",
    userprincipalname="godwin@foobar.onmicrosoft.com",
)

#### Using file paths and userprincipalname(org provided email) of user
documents = loader.load_data(
    file_ids=[
        "subfolder/subfolder2/fileid1.pdf",
        "subfolder/subfolder3/fileid2.docx",
    ],
    userprincipalname="godwin@foobar.onmicrosoft.com",
)
```

#### Author

[Godwin Paul Vincent](https://github.com/godwin3737)

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
