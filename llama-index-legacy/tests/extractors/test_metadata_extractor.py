"""Test dataset generation."""

from tempfile import TemporaryDirectory

from llama_index.legacy import SimpleDirectoryReader
from llama_index.legacy.extractors import (
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.legacy.ingestion import IngestionPipeline
from llama_index.legacy.llms import MockLLM
from llama_index.legacy.text_splitter import TokenTextSplitter

test_data = """
# High-Level Concepts

This is a quick guide to the high-level concepts you'll encounter frequently when building LLM applications.

```{tip}
If you haven't, [install LlamaIndex](/getting_started/installation.md) and complete the [starter tutorial](/getting_started/starter_example.md) before you read this. It will help ground these steps in your experience.
```

## Retrieval Augmented Generation (RAG)

LLMs are trained on enormous bodies of data but they aren't trained on **your** data. Retrieval-Augmented Generation (RAG) solves this problem by adding your data to the data LLMs already have access to. You will see references to RAG frequently in this documentation.

In RAG, your data is loaded and prepared for queries or "indexed". User queries act on the index, which filters your data down to the most relevant context. This context and your query then go to the LLM along with a prompt, and the LLM provides a response.

Even if what you're building is a chatbot or an agent, you'll want to know RAG techniques for getting data into your application.

![](/_static/getting_started/basic_rag.png)

## Stages within RAG

There are five key stages within RAG, which in turn will be a part of any larger application you build. These are:

- **Loading**: this refers to getting your data from where it lives -- whether it's text files, PDFs, another website, a database, or an API -- into your pipeline. [LlamaHub](https://llamahub.ai/) provides hundreds of connectors to choose from.

- **Indexing**: this means creating a data structure that allows for querying the data. For LLMs this nearly always means creating `vector embeddings`, numerical representations of the meaning of your data, as well as numerous other metadata strategies to make it easy to accurately find contextually relevant data.

- **Storing**: once your data is indexed you will almost always want to store your index, as well as other metadata, to avoid having to re-index it.

- **Querying**: for any given indexing strategy there are many ways you can utilize LLMs and LlamaIndex data structures to query, including sub-queries, multi-step queries and hybrid strategies.

- **Evaluation**: a critical step in any pipeline is checking how effective it is relative to other strategies, or when you make changes. Evaluation provides objective measures of how accurate, faithful and fast your responses to queries are.

![](/_static/getting_started/stages.png)

## Important concepts within each step

There are also some terms you'll encounter that refer to steps within each of these stages.
"""


def test_metadata_extractor() -> None:
    """Test metadata extraction."""
    llm = MockLLM()

    with TemporaryDirectory() as tmp_dir:
        with open(f"{tmp_dir}/test.md", "w") as f:
            f.write(test_data)

        docs = SimpleDirectoryReader(
            tmp_dir, recursive=True, required_exts=[".md"]
        ).load_data()

        text_splitter = TokenTextSplitter(
            separator=" ", chunk_size=64, chunk_overlap=16
        )

        extractors = [
            TitleExtractor(nodes=3, llm=llm),
            QuestionsAnsweredExtractor(questions=2, llm=llm),
        ]

        transformations = [text_splitter, *extractors]

        pipeline = IngestionPipeline(transformations=transformations)

        nodes = pipeline.run(documents=docs)

        assert (
            nodes[0].metadata["document_title"] != nodes[-1].metadata["document_title"]
        )
