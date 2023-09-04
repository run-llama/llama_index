from llama_index.indices.prompt_helper import PromptHelper
from llama_index.indices.service_context import ServiceContext
from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.llms import MockLLM
from llama_index.token_counter.mock_embed_model import MockEmbedding


def test_service_context_serialize() -> None:
    text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)

    metadata_extractor = MetadataExtractor(
        extractors=[
            SummaryExtractor(),
            QuestionsAnsweredExtractor(),
            TitleExtractor(),
        ]
    )

    node_parser = SimpleNodeParser.from_defaults(
        text_splitter=text_splitter, metadata_extractor=metadata_extractor
    )

    llm = MockLLM(max_tokens=1)
    embed_model = MockEmbedding(embed_dim=1)

    prompt_helper = PromptHelper(context_window=1)

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
        prompt_helper=prompt_helper,
    )

    service_context_dict = service_context.to_dict()

    assert service_context_dict["llm"]["max_tokens"] == 1
    assert service_context_dict["embed_model"]["embed_dim"] == 1
    assert service_context_dict["text_splitter"]["chunk_size"] == 1
    assert len(service_context_dict["extractors"]) == 3
    assert service_context_dict["prompt_helper"]["context_window"] == 1

    loaded_service_context = ServiceContext.from_dict(service_context_dict)

    assert isinstance(loaded_service_context.llm, MockLLM)
    assert isinstance(loaded_service_context.embed_model, MockEmbedding)
    assert isinstance(loaded_service_context.node_parser, SimpleNodeParser)
    assert isinstance(loaded_service_context.prompt_helper, PromptHelper)
    assert isinstance(
        loaded_service_context.node_parser.text_splitter, TokenTextSplitter
    )

    assert loaded_service_context.node_parser.metadata_extractor is not None
    assert len(loaded_service_context.node_parser.metadata_extractor.extractors) == 3
    assert loaded_service_context.node_parser.text_splitter.chunk_size == 1
    assert loaded_service_context.prompt_helper.context_window == 1
    assert loaded_service_context.llm.max_tokens == 1
    assert loaded_service_context.embed_model.embed_dim == 1
