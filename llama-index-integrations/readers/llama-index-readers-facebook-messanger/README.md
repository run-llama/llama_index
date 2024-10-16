# LlamaIndex Readers Integration: Facebook_Messanger
# Messanger chat loader

```bash
pip install llama-index-readers-messenger
```

## Export a Facebook Messanger chat

1. Go to Facebook's Download Your Information (https://www.facebook.com/login.php?next=https%3A%2F%2Fwww.facebook.com%2Fdyi%2F) page.
2. Select Messages and click on Request a download.
3. Once the file is ready, download the file and unzip it.
4. Find the chat you want to analyze in the messages folder, typically in .json format.
5. Save the .json file in your working directory.

For more info see [Meta's Help Center](https://www.facebook.com/help/1701730696756992/)

## Usage

- Messages will get saved in the format: `{timestamp} {author}: {message}`.This is helpful when analyzing conversations, particularly in group chats, and allows filtering based on users, dates, or keywords.
- Metadata automatically included: `source` (file name), `author` and `timestamp`.

```python
from pathlib import Path

from llama_index.readers.messenger import FacebookMessengerLoader

path = "facebook_chat.json"
loader = FacebookMessengerLoader(path=path)
documents = loader.load_data()

# see what's created
documents[0]
# >>> Document(text='2023-02-20 00:00:00 Jane Doe: Hello, how are you?', doc_id='b7a2d508-3ba2-42e1-a3bc-8bf235232364', embedding=None, extra_info={'source': 'Facebook Chat with Jane Doe', 'author': 'Jane Doe', 'timestamp': '2023-02-20 00:00:00'})
```

This loader is designed to be used as a way to load Facebook Messenger data into https://github.com/run-llama/llama_index/