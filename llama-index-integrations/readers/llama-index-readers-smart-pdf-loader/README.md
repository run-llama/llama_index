# Smart PDF Loader

```bash
pip install llama-index-readers-smart-pdf-loader
```

SmartPDFLoader is a super fast PDF reader that understands the layout structure of PDFs such as nested sections, nested lists, paragraphs and tables.
It uses layout information to smartly chunk PDFs into optimal short contexts for LLMs.

## Requirements

Install the llmsherpa library if it is not already present:

```
pip install llmsherpa
```

## Usage

Here's an example usage of the SmartPDFLoader:

```python
from llama_index.readers.smart_pdf_loader import SmartPDFLoader

llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
pdf_url = "https://arxiv.org/pdf/1910.13461.pdf"  # also allowed is a file path e.g. /home/downloads/xyz.pdf
pdf_loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)
documents = pdf_loader.load_data(pdf_url)
```

Now you can use the documents with other LlamaIndex components. For example, for retrieval augmented generation, try this:

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("list all the tasks that work with bart")
print(response)

response = query_engine.query("what is the bart performance score on squad")
print(response)
```

## More Examples

SmartPDFLoader is based on LayoutPDFReader from [llmsherpa](https://github.com/nlmatics/llmsherpa) library. See the [documentation](<(https://github.com/nlmatics/llmsherpa)>) there to explore other ways to use the library for connecting data from your PDFs with LLMs.

- [Summarize a section using prompts](https://github.com/nlmatics/llmsherpa#summarize-a-section-using-prompts)
- [Analyze a table using prompts](https://github.com/nlmatics/llmsherpa#analyze-a-table-using-prompts)
- [Vector search and RAG](https://github.com/nlmatics/llmsherpa#vector-search-and-retrieval-augmented-generation-with-smart-chunking)
