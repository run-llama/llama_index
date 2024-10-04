import os
import pytest

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.octoai import OctoAI


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in OctoAI.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


@pytest.mark.skipif(
    os.environ.get("OCTOAI_TOKEN") is None, reason="OCTOAI_TOKEN env var not set"
)
def test_completion():
    octoai = OctoAI(token=os.getenv("OCTOAI_TOKEN"))
    assert octoai.complete("Who is Paul Graham?")


@pytest.mark.skipif(
    os.environ.get("OCTOAI_TOKEN") is None, reason="OCTOAI_TOKEN env var not set"
)
def test_chat():
    octoai = OctoAI(token=os.getenv("OCTOAI_TOKEN"))
    assert octoai.chat(
        [ChatMessage(role=MessageRole.USER, content="Who is Paul Graham?")]
    )
