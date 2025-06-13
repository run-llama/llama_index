import pytest

from typing import Dict
from llama_index.core.voice_agents.events import ConversationBaseEvent


@pytest.fixture()
def json_event() -> Dict[str, str]:
    return {"type": "text"}


def test_event_serialization(json_event: Dict[str, str]) -> None:
    assert ConversationBaseEvent(type_t="text").model_dump(by_alias=True) == json_event
