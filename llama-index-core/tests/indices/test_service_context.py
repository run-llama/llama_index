from typing import List

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.extractors import (
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.llms.mock import MockLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TransformComponent
from llama_index.core.service_context import ServiceContext


def test_service_context_serialize() -> None:
    extractors: List[TransformComponent] = [
        SummaryExtractor(),
        QuestionsAnsweredExtractor(),
        TitleExtractor(),
    ]

    node_parser = SentenceSplitter(chunk_size=1, chunk_overlap=0)

    transformations: List[TransformComponent] = [node_parser, *extractors]

    llm = MockLLM(max_tokens=1)
    embed_model = MockEmbedding(embed_dim=1)

    prompt_helper = PromptHelper(context_window=1)

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        transformations=transformations,
        prompt_helper=prompt_helper,
    )

    service_context_dict = service_context.to_dict()

    assert service_context_dict["llm"]["max_tokens"] == 1
    assert service_context_dict["embed_model"]["embed_dim"] == 1
    assert service_context_dict["prompt_helper"]["context_window"] == 1

    loaded_service_context = ServiceContext.from_dict(service_context_dict)

    assert isinstance(loaded_service_context.llm, MockLLM)
    assert isinstance(loaded_service_context.embed_model, MockEmbedding)
    assert isinstance(loaded_service_context.transformations[0], SentenceSplitter)
    assert isinstance(loaded_service_context.prompt_helper, PromptHelper)

    assert len(loaded_service_context.transformations) == 4
    assert loaded_service_context.transformations[0].chunk_size == 1
    assert loaded_service_context.prompt_helper.context_window == 1
    assert loaded_service_context.llm.max_tokens == 1
    assert loaded_service_context.embed_model.embed_dim == 1
