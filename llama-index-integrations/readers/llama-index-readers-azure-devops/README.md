# LlamaIndex Readers Integration: Azure Devops

`pip install llama-index-readers-azure-devops`

The Azure Devops readers package enables you to read files from your azure devops repositories

The reader will require a personal access token (which you can generate under your account settings).

## Usage

This reader will read through a repo, with options to specifically filter directories and file extensions.

Here is an example of how to use it

```python
from llama_index.readers.azure_devops import AzureDevopsReader

az_devops_loader = AzureDevopsLoader(
    access_token="<your-access-token>",
    organization_name="<organization-name>",
    project_name="<project-name>",
    repo="<repository-name>",
    file_filter=lambda file_path: file_path.endswith(".py"),
)  # Optional: you can provide any callable that returns a boolean to filter files of your choice

documents = az_devops_loader.load_data(
    folder="<folder-path>",  # The folder to load documents from, defaults to root.
    branch="<branch-name>",
)  # The branch to load documents from, defaults to head of the repo
```
