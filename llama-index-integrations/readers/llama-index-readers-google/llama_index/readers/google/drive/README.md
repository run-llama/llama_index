# Google Drive Loader

This loader reads files from Google Drive using folder or file ids. To use this loader, you need to pass in a list of file id's or folder id.

### folder_id

You can extract a folder_id directly from its drive URL.

For example, the folder_id of `https://drive.google.com/drive/folders/1w7XryYu6mL9VLmfyqUkA4_fRnDbsCqV-` is `1w7XryYu6mL9VLmfyqUkA4_fRnDbsCqV-`.

### file_id

You can extract a file_id directly from its sharable drive URL.

For example, the file_id of `https://drive.google.com/file/d/1LEqD_zQiOizKrBKZYKJtER_h6i49wE-y/view?usp=sharing` is `1LEqD_zQiOizKrBKZYKJtER_h6i49wE-y`.

### mime_types

You can also filter the files by the mimeType e.g.: `mime_types=["application/vnd.google-apps.document"]`

## Usage

We need `credentials.json` file to use this reader.

1. You need to create a service account folllowing the steps mentioned [here](https://cloud.google.com/iam/docs/keys-create-delete)
2. Get your json file and rename to `credentials.json` and move to the project root

> Note: If you are not using Google Workspaces (formerly GSuite), You'll need to share your document making it public, or inviting your service account as an reader/editor of the folder or file.

Finally, make sure you enable "Google Drive API" in the console of your Google App.

```python
from llama_index import download_loader

GoogleDriveReader = download_loader("GoogleDriveReader")

loader = GoogleDriveReader()

#### Using folder id
documents = loader.load_data(folder_id="folderid")

#### Using file ids
documents = loader.load_data(file_ids=["fileid1", "fileid2"])
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.
