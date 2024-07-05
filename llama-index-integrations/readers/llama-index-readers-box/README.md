# LlamaIndex Readers Integration: Box

```bash
pip install llama-index-readers-box
```

### Authentication

#### Client credential gran (CCG)

Create a new application in the Box Developer Console and generate a new client ID and client secret.
Create a .env file with the following content:

```bash
# Common settings
BOX_CLIENT_ID = YOUR_CLIENT_ID
BOX_CLIENT_SECRET = YOUR_CLIENT_SECRET

# CCG Settings
BOX_ENTERPRISE_ID = YOUR_BOX_ENTERPRISE_ID
BOX_USER_ID = YOUR_BOX_USER_ID (optional)
```

By default the CCG client will use a service account associated with the application. Depending on how the files are shared, the service account may not have access to all the files.

If you want to use a different user, you can specify the user ID in the .env file. In this case make sure your application can impersonate and/or generate user tokens in the scope.

Checkout this guide for more information on how to setup the CCG: [Box CCG Guide](https://developer.box.com/guides/authentication/client-credentials/)

#### JSON web tokens (JWT)

## Usage

### Box Reader

This is a simple reader that can be used to load files from Box.
It uses the LLamaIndex simple reader, and does not take advantage of any Box specific features.

```python
reader = BoxReader(
    box_client_id=box_config.client_id,
    box_client_secret=box_config.client_secret,
    box_enterprise_id=box_config.enterprise_id,
    box_user_id=box_config.ccg_user_id,
)
docs = box_reader.load_data(file_ids=["file_id1", "file_id2"])
```

#### file_ids

List of file ids to load, should be a list of strings.

#### folder_id

Folder id to load, should be a string.

### Box Reader Text Extraction

> Future implementation

### Box Reader AI Prompt

> Future implementation

### Box Reader AI Extraction

> Future implementation

#### Author

[Box-Community](https://github.com/box-community)
This is an open source reader, contributions are welcome.

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
