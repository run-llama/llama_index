import os
import pytest

from llama_index import ServiceContext
from llama_index.embeddings import OpenAIEmbedding
from llama_index.indices import VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index.node_parser.text import TokenTextSplitter
from llama_index.schema import Document

def setup_module() -> None:
    os.environ["PLATFORM_AUTO_UPLOAD"] = "true"
    os.environ["OPENAI_API_KEY"] = "sk-" + ("a" * 48)


def teardown_module() -> None:
    del os.environ["PLATFORM_AUTO_UPLOAD"]
    del os.environ["OPENAI_API_KEY"]


@pytest.mark.integration()
def test_service_context_register() -> None:
    llm = OpenAI()
    embed_model = OpenAIEmbedding()

    service_context = ServiceContext.from_defaults(
        llm=llm, 
        embed_model=embed_model,
        text_splitter=TokenTextSplitter(chunk_size=100),
    )

    index = VectorStoreIndex.from_documents(
        [],
        service_context=service_context,
        remote_pipeline_name="test_pipeline"
    )

    assert isinstance(index.service_context.node_parser, TokenTextSplitter)
    assert index.service_context.node_parser.chunk_size == 100

    new_index = VectorStoreIndex.from_documents(
        [],
        from_pipeline_name="test_pipeline"
    )

    assert isinstance(new_index.service_context.node_parser, TokenTextSplitter)
    assert new_index.service_context.node_parser.chunk_size == 100
