# LlamaIndex Readers Integration: Box

```bash
pip install llama-index-readers-box
```

### Authentication

#### Client credential gran (CCG)

Create a new application in the Box Developer Console and generate a new client ID and client secret.
Create a .env file with the following content:

```bash
# CCG settings
BOX_CLIENT_ID = YOUR_CLIENT_ID
BOX_CLIENT_SECRET = YOUR_CLIENT_SECRET

# Common Settings
BOX_ENTERPRISE_ID = YOUR_BOX_ENTERPRISE_ID
BOX_USER_ID = YOUR_BOX_USER_ID (optional)
```

By default the CCG client will use a service account associated with the application. Depending on how the files are shared, the service account may not have access to all the files.

If you want to use a different user, you can specify the user ID in the .env file. In this case make sure your application can impersonate and/or generate user tokens in the scope.

Checkout this guide for more information on how to setup the CCG: [Box CCG Guide](https://developer.box.com/guides/authentication/client-credentials/)

#### JSON web tokens (JWT)

Create a new application in the Box Developer Console and generate a new `.config.json` file.
Create a .env file with the following content:

```bash
# Common settings
BOX_ENTERPRISE_ID = 877840855
BOX_USER_ID = 18622116055

# JWT Settings
JWT_CONFIG_PATH = /path/to/your/.config.json
```

By default the JWT client will use a service account associated with the application. Depending on how the files are shared, the service account may not have access to all the files.

If you want to use a different user, you can specify the user ID in the .env file. In this case make sure your application can impersonate and/or generate user tokens in the scope.

Checkout this guide for more information on how to setup the JWT: [Box JWT Guide](https://developer.box.com/guides/authentication/jwt/jwt-setup/)

## Usage

### Box Reader

This is a simple reader that can be used to load files from Box.
It uses the LLamaIndex simple reader, and does not take advantage of any Box specific features.

#### Using CCG authentication

```python
from box_sdk_gen import CCGConfig

ccg_conf = CCGConfig(
    client_id="your_client_id",
    client_secret="your_client_secret",
    enterprise_id="your_enterprise_id",
    user_id="your_ccg_user_id",  # optional
)

reader = BoxReader(box_config=ccg_conf)
docs = reader.load_data(file_ids=["file_id1", "file_id2"])
```

#### Using JWT authentication

```python
from box_sdk_gen import JWTConfig

jwt_conf = JWTConfig.from_config_file(jwt_config_path)
user_id = "your_user_id"  # optional
if user_id:
    jwt.user_id = user_id
    jwt.enterprise_id = None

reader = BoxReader(box_config=jwt_conf)
docs = reader.load_data(file_ids=["file_id1", "file_id2"])
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
