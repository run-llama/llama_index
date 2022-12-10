"""Test PromptHelper."""
from typing import List
from unittest.mock import patch

from gpt_index.indices.data_structs import Node
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.prompts.base import Prompt


def mock_tokenizer(text: str) -> List[str]:
    """Mock tokenizer."""
    return text.split(" ")


@patch.object(PromptHelper, "_tokenizer", mock_tokenizer)
def test_get_text_from_nodes():
    """Test get_text_from_nodes."""
    # test prompt uses up one token
    test_prompt_txt = "test{text}"
    test_prompt = Prompt(input_variables=["text"], template=test_prompt_txt)
    # set max_input_size=5
    prompt_helper = PromptHelper(max_input_size=5, num_output=0, max_chunk_overlap=0)
    node1 = Node(text="This is a test test2 test3")
    node2 = Node(text="Hello world world2 world3")

    response = prompt_helper.get_text_from_nodes([node1, node2], prompt=test_prompt)
    assert response == ("This is a test\n" "Hello world world 2 world3")
