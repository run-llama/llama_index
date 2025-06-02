# Google Chat Loader

`pip install llama-index-readers-google`

This loader takes in IDs of Google Chat spaces or messages and parses the chat history into `Document`s. The space/message ID can be found in the URL, as shown below:

- mail.google.com/chat/u/0/#chat/space/**\<CHAT_ID\>**

Before using this loader, you need to create a Google Cloud Platform (GCP) project with a Google Workspace account. Then, you need to authorize the app with user credentials. Follow the prerequisites and steps 1 and 2 of [this guide](https://developers.google.com/workspace/chat/authenticate-authorize-chat-user). After downloading the client secret JSON file, rename it as **`credentials.json`** and save it into your project folder.

## Usage

To use this loader, pass in an array of Google Chat IDs.

```py
from llama_index.readers.google import GoogleChatReader

space_names = ["<CHAT_ID>"]
chatReader = GoogleChatReader()
docs = chatReader.load_data(space_names=space_names)
```

There are also additional parameters that allow you to specify which chat messages you want to read:

- `num_messages`: The number of messages to load (may not be exact). If `order_asc` is True, then loads `num_messages` from the beginning of the chat. If `order_asc` is False, then loads `num_messages` from the end of the chat.
- `after`: Only loads messages after this timestamp (a datetime object)
- `before`: Only loads messages before this timestamp (a datetime object)
- `order_asc`: If True, then orders messages in ascending order. Otherwise orders messages in descending order.

## Examples

```py
from llama_index.readers.google import GoogleChatReader
from llama_index.core import VectorStoreIndex

space_names = ["<CHAT_ID>"]
chatReader = GoogleChatReader()
docs = chatReader.load_data(space_names=space_names)
index = VectorStoreIndex.from_documents(docs)
query_eng = index.as_query_engine()
print(query_eng.query("What was this conversation about?"))
```
