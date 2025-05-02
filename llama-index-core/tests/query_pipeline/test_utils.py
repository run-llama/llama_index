from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.query_pipeline.query import validate_and_convert_stringable


def test_validate_and_convert_stringable() -> None:
    """Test conversion of stringable object into string."""
    message = ChatMessage(role=MessageRole.USER, content="hello")
    assert validate_and_convert_stringable(message) == "user: hello"
