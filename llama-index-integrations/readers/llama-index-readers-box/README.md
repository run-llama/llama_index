# LlamaIndex: Box Readers

This open-source integration brings the capabilities of Box.com to the LLama-Index, empowering developers building Retrieval Augmented Generation (RAG) and other LLM applications.

This README will guide you through installation, usage, and explore the functionalities of each reader.

## Installation

```bash
pip install llama-index-readers-box
```

## Available readers

We provide multiple readers, including:

- [Box Reader](llama_index/readers/box/BoxReader/README.md) - Implementation of the SimpleReader interface to read files from Box.
- [Box Text Extraction](llama_index/readers/box/BoxReaderTextExtraction/README.md) - Uses Box text representation to extract text from document.
- [Box AI Prompt](llama_index/readers/box/BoxReaderAIPrompt/README.md) - Uses Box AI to extract context from documents
- [Box AI Extraction](llama_index/readers/box/BoxReaderAIExtraction/README.md) - Uses Box AI to extract structured data from documents

> [!IMPORTANT]
> Box AI features are only available to E+ customers.

## Authentication

### Client credential gran (CCG)

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

### JSON web tokens (JWT)

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

> [!WARNING]
> The JWT authentication requires extra dependencies in the SDK. You can install them by running:

```bash
pip install "box-sdk-gen[jwt]"
```

## Box Client

To work with the box readers, you will need to provide a Box Client.
The Box Client can be created using either the Client Credential Grant (CCG), JSON Web Tokens (JWT), OAuth 2.0, and developer token.

### Using CCG authentication

```python
from box_sdk_gen import CCGConfig, BoxCCGAuth, BoxClient

config = CCGConfig(
    client_id="your_client_id",
    client_secret="your_client_secret",
    enterprise_id="your_enterprise_id",
    user_id="your_ccg_user_id",  # Optional
)
auth = BoxCCGAuth(config)
if config.user_id:
    auth.with_user_subject(config.user_id)
client = BoxClient(auth)

reader = BoxReader(box_client=client)
```

### Using JWT authentication

```python
from box_sdk_gen import JWTConfig, BoxJWTAuth, BoxClient

# Using manual configuration
config = JWTConfig(
    client_id="YOUR_BOX_CLIENT_ID",
    client_secret="YOUR_BOX_CLIENT_SECRET",
    jwt_key_id="YOUR_BOX_JWT_KEY_ID",
    private_key="YOUR_BOX_PRIVATE_KEY",
    private_key_passphrase="YOUR_BOX_PRIVATE_KEY_PASSPHRASE",
    enterprise_id="YOUR_BOX_ENTERPRISE_ID",
    user_id="YOUR_BOX_USER_ID",
)

# Using configuration file
config = JWTConfig.from_config_file("path/to/your/.config.json")


user_id = "1234"
if user_id:
    config.user_id = user_id
    config.enterprise_id = None
auth = BoxJWTAuth(config)
client = BoxClient(auth)

reader = BoxReader(box_client=client)
```

## Author

[Box-Community](https://github.com/box-community)
This is an open source reader, contributions are welcome.

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
