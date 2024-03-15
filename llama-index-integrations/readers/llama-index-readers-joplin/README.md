# Joplin (Markdown) Loader

```bash
pip install llama-index-readers-joplin
```

> [Joplin](https://joplinapp.org/) is an open source note-taking app. Capture your thoughts and securely access them from any device.

This readme covers how to load documents from a `Joplin` database.

`Joplin` has a [REST API](https://joplinapp.org/api/references/rest_api/) for accessing its local database. This reader uses the API to retrieve all notes in the database and their metadata. This requires an access token that can be obtained from the app by following these steps:

1. Open the `Joplin` app. The app must stay open while the documents are being loaded.
2. Go to settings / options and select "Web Clipper".
3. Make sure that the Web Clipper service is enabled.
4. Under "Advanced Options", copy the authorization token.

You may either initialize the reader directly with the access token, or store it in the environment variable JOPLIN_ACCESS_TOKEN.

An alternative to this approach is to export the `Joplin`'s note database to Markdown files (optionally, with Front Matter metadata) and use a Markdown reader, such as ObsidianReader, to load them.

## Usage

Here's an example usage of the JoplinReader.

```python
import os

from llama_index.readers.joplin import JoplinReader

documents = JoplinReader(
    access_token="<access_token>"
).load_data()  # Returns list of documents
```
