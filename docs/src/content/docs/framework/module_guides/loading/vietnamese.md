---
title: Vietnamese PDF Ingestion
---

Vietnamese words are often several syllables with spaces between them (`công nghệ`, `năng lượng mặt trời`). [`VectorStoreIndex.from_documents()`](/python/framework/module_guides/indexing/vector_store_index) chunks with [`SentenceSplitter`](/python/framework/module_guides/loading/node_parsers/modules#sentencesplitter) before it embeds. That splitter prefers sentences, then splits long spans on spaces. The text inside each chunk is what gets embedded, so syllable boundaries become the tokens the retriever sees.

Word-segment the text first, then chunk. On a corpus of Vietnamese solar technical PDFs (MOIT/EVN), segmentation before chunking improved retrieval precision by about 15% compared with chunking the raw extract. That figure is from that corpus ([#23709](https://github.com/run-llama/llama_index/issues/23709)). Measure again on your own documents.

Tone marks are part of the word (`ma`, `má`, `mà`, `mả`, `mã`, `mạ`). Extract Vietnamese PDFs with pypdf so those marks stay in the string you segment and embed.

## Install

```bash
pip install llama-index llama-index-readers-file underthesea
```

`llama-index-readers-file` loads `.pdf` files and depends on `pypdf`. [`PDFReader`](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/docs/base.py) is the reader `SimpleDirectoryReader` uses for `.pdf`. [underthesea](https://pypi.org/project/underthesea/) supplies `word_tokenize`.

`VectorStoreIndex.from_documents()` embeds with `Settings.embed_model`. With the `llama-index` package installed, that default is OpenAI, which reads `OPENAI_API_KEY`.

## Extract text with pypdf

Keep the default `PDFReader` for Vietnamese PDFs. It extracts text with pypdf. pdfminer (including `pdfminer.six`) has been reported to drop tone marks on these files ([#23709](https://github.com/run-llama/llama_index/issues/23709)). The string passed to `word_tokenize` and to the embedding model is then a different word.

When you pass `file_extractor`, point `.pdf` at a `PDFReader` instance. That mapping replaces the default reader for the extensions you list.

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader

documents = SimpleDirectoryReader(
    input_dir="./solar_docs_vi",
    required_exts=[".pdf"],
    file_extractor={".pdf": PDFReader()},
).load_data()
```

Each page becomes its own `Document`, with `page_label` and `file_name` in `metadata`. `PDFReader(return_full_document=True)` joins every page into one document.

`PDFReader` returns the embedded text layer. A scanned page with no text layer stays empty; run OCR before this step.

## Segment words before chunking

`word_tokenize(..., format="text")` joins the syllables of each word with `_` and leaves a space between words:

```text
công nghệ -> công_nghệ
```

`from_documents()` runs `Settings.transformations` on the document text. The default transformation is `SentenceSplitter`. Segment before that call.

`Document.text` is a read-only view of the content. Write the segmented string with `set_content()`. Metadata such as `page_label` stays on the document.

```python
from llama_index.core import (
    QueryBundle,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.readers.file import PDFReader
from underthesea import word_tokenize

documents = SimpleDirectoryReader(
    input_dir="./solar_docs_vi",
    required_exts=[".pdf"],
    file_extractor={".pdf": PDFReader()},
).load_data()

for doc in documents:
    if not doc.text.strip():
        continue
    doc.set_content(word_tokenize(doc.text, format="text"))

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
question = "Công suất lắp đặt điện mặt trời là bao nhiêu?"
response = query_engine.query(
    QueryBundle(
        query_str=question,
        custom_embedding_strs=[word_tokenize(question, format="text")],
    )
)
print(response)
```

The indexed text contains tokens such as `mặt_trời` (`word_tokenize("điện mặt trời", format="text")` is `điện mặt_trời`). `QueryBundle.custom_embedding_strs` embeds the segmented question, and `query_str` is the sentence the LLM answers. A query that is still split into syllables does not match those word tokens.

underthesea may split punctuation into its own token (`Việt_Nam .`). The word itself stays one whitespace token, which is what `SentenceSplitter` needs.

Skip blank pages. `word_tokenize` on an empty string returns `""`.

## Same step in an ingestion pipeline

A pipeline applies transformations in order. Put segmentation in front of `SentenceSplitter`. This replaces the loop above; doing both segments the text twice.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TransformComponent
from underthesea import word_tokenize


class VietnameseWordSegmenter(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            text = node.get_content()
            if text.strip():
                node.set_content(word_tokenize(text, format="text"))
        return nodes


pipeline = IngestionPipeline(
    transformations=[
        VietnameseWordSegmenter(),
        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
    ]
)
nodes = pipeline.run(documents=documents)
index = VectorStoreIndex(nodes)
```

`from_documents()` accepts the same list through `transformations`:

```python
index = VectorStoreIndex.from_documents(
    documents,
    transformations=[
        VietnameseWordSegmenter(),
        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
    ],
)
```

Use the unsegmented `documents` from `SimpleDirectoryReader` with this list. The segmenter runs first, then the splitter.
