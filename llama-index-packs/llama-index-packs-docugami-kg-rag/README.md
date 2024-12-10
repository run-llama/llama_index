# Docugami KG-RAG Pack

This LlamaPack provides an end-to-end Knowledge Graph Retrieval Augmented Generation flow using Docugami.

## Process Documents in Docugami (before you use this template)

Before you use this llamapack, you must have some documents already processed in Docugami. Here's what you need to get started:

1. Create a [Docugami workspace](https://app.docugami.com/) (free trials available)
1. Create an access token via the Developer Playground for your workspace. [Detailed instructions](https://help.docugami.com/home/docugami-api).
1. Add your documents to Docugami for processing. There are two ways to do this:
   - Upload via the simple Docugami web experience. [Detailed instructions](https://help.docugami.com/home/adding-documents).
   - Upload via the Docugami API, specifically the [documents](https://api-docs.docugami.com/#tag/documents/operation/upload-document) endpoint. Code samples are available for python and JavaScript or you can use the [docugami](https://pypi.org/project/docugami/) python library.

Once your documents are in Docugami, they are processed and organized into sets of similar documents, e.g. NDAs, Lease Agreements, and Service Agreements. Docugami is not limited to any particular types of documents, and the clusters created depend on your particular documents. You can [change the docset assignments](https://help.docugami.com/home/working-with-the-doc-sets-view) later if you wish. You can monitor file status in the simple Docugami webapp, or use a [webhook](https://api-docs.docugami.com/#tag/webhooks) to be informed when your documents are done processing.

## Environment Variables

You need to set some required environment variables before using your new app based on this template. These are used to index as well as run the application, and exceptions are raised if the following required environment variables are not set:

1. `OPENAI_API_KEY`: from the OpenAI platform.
1. `DOCUGAMI_API_KEY`: from the [Docugami Developer Playground](https://help.docugami.com/home/docugami-api)

```shell
export OPENAI_API_KEY=...
export DOCUGAMI_API_KEY=...
```

## Using the llamapack

Once your documents are finished processing, you can build and use the agent by adding the following code

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
DocugamiKgRagPack = download_llama_pack(
    "DocugamiKgRagPack", "./docugami_kg_rag"
)

docset_id = ...
pack = DocugamiKgRagPack()
pack.build_agent_for_docset(docset_id)
pack.run(...)
```
