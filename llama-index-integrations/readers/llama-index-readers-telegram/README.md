# Telegram Loader

```bash
pip install llama-index-readers-telegram
```

This loader fetches posts/chat messages/comments from Telegram channels or chats into `Document`s.

Before working with Telegram’s API, you need to get your own API ID and hash:

1. [Login to your Telegram account](https://my.telegram.org) with the phone number of the developer account to use.
2. Click under API Development tools.
3. A Create new application window will appear. Fill in your application details. There is no need to enter any URL, and only the first two fields (App title and Short name) can currently be changed later.
4. Click on Create application at the end. Remember that your API hash is secret and Telegram won’t let you revoke it. Don’t post it anywhere!

This API ID and hash is the one used by your application, not your phone number. You can use this API ID and hash with any phone number.

## Usage

The first parameter you pass to the constructor of the TelegramReader is the session_name, and defaults to be the session name (or full path). That is, if you create a TelegramReader(session_name='anon', ...) instance and run load_data(), an `anon.session` file will be created in the working directory where you run this loader.

The Auth procedure asks for:

- Security Code
- Password

```bash
Please enter the code you received: 12345
Please enter your password: *******
(You are now logged in)
```

If the `.session` file already existed, it will not login again, so be aware of this if you move or rename the file! See [here](https://docs.telethon.dev/en/stable/index.html) for more instructions.

To use this loader, you simply need to pass in a entity name.

```python
from llama_index.readers.telegram import TelegramReader

loader = TelegramReader(
    session_name="[YOUR_SESSION_NAME]",
    api_id="[YOUR_API_ID]",
    api_hash="[YOUR_API_HASH]",
    phone_number="[YOUR_PHONE_NUMBER]",
)
documents = loader.load_data(
    entity_name="https://t.me/[ENTITY_NAME]", limit=100
)
```

## Examples

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
