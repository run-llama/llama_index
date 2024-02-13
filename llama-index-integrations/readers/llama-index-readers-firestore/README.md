# Firestore Loader

This loader loads from a Firestore collection or a specific document from Firestore. The loader assumes your project already has the google cloud credentials loaded. To find out how to set up credentials, [see here](https://cloud.google.com/docs/authentication/provide-credentials-adc).

## Usage

To initialize the loader, provide the project-id of the google cloud project.

## Initializing the reader

```python
from llama_index import download_loader

FirestoreReader = download_loader("FirestoreReader")
reader = FirestoreReader(project_id="<Your Project ID>")
```

## Loading Data from a Firestore Collection

Load data from a Firestore collection with the load_data method:
The collection path should include all previous documents and collections if it is a nested collection.

```python
documents = reader.load_data(collection="foo/bar/abc/")
```

## Loading a Single Document from Firestore

Load a single document from Firestore with the load_document method:

```python
document = reader.load_document(document_url="foo/bar/abc/MY_DOCUMENT")
```

Note: load_data returns a list of Document objects, whereas load_document returns a single Document object.

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.
