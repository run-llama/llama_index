import pytest
from unittest.mock import patch, MagicMock
from typing import Sequence

from llama_index.core.schema import ImageDocument, ImageNode
from llama_index.core.base.llms.types import ImageBlock


class TestDashScopeMultiModalInputParameters:
    """Test _get_input_parameters handles ImageDocument correctly."""

    def test_image_document_with_url(self):
        """Test that ImageDocument with image_url is handled correctly."""
        from llama_index.multi_modal_llms.dashscope.base import DashScopeMultiModal

        with patch.object(DashScopeMultiModal, "__init__", lambda x: None):
            llm = DashScopeMultiModal()
            llm.model_name = "qwen-vl-max"
            llm.incremental_output = True
            llm.top_k = None
            llm.top_p = None
            llm.seed = 1234

            # Create mock ImageDocument with image_url
            with patch("llama_index.core.schema.is_image_url_pil", return_value=True):
                image_doc = ImageDocument(image_url="https://example.com/image.jpg")

            message, params = llm._get_input_parameters(
                "What is this?", [image_doc]
            )

            # Verify the message content includes the image URL
            content = message.content
            assert isinstance(content, list)
            assert any(
                "image" in item and "https://example.com/image.jpg" in str(item["image"])
                for item in content
            ), f"Expected image URL in content, got: {content}"

    def test_image_node_is_converted(self):
        """Test that ImageNode is properly converted to ImageBlock."""
        from llama_index.multi_modal_llms.dashscope.base import DashScopeMultiModal

        with patch.object(DashScopeMultiModal, "__init__", lambda x: None):
            llm = DashScopeMultiModal()
            llm.model_name = "qwen-vl-max"
            llm.incremental_output = True
            llm.top_k = None
            llm.top_p = None
            llm.seed = 1234

            # Create ImageNode with URL
            image_node = ImageNode(image_url="https://example.com/image.jpg")

            message, params = llm._get_input_parameters(
                "What is this?", [image_node]
            )

            content = message.content
            assert isinstance(content, list)
            assert any("image" in item for item in content)

    def test_image_block_is_used_directly(self):
        """Test that ImageBlock is used directly without conversion."""
        from llama_index.multi_modal_llms.dashscope.base import DashScopeMultiModal

        with patch.object(DashScopeMultiModal, "__init__", lambda x: None):
            llm = DashScopeMultiModal()
            llm.model_name = "qwen-vl-max"
            llm.incremental_output = True
            llm.top_k = None
            llm.top_p = None
            llm.seed = 1234

            # Create ImageBlock directly
            image_block = ImageBlock(url="https://example.com/image.jpg")

            message, params = llm._get_input_parameters(
                "What is this?", [image_block]
            )

            content = message.content
            assert isinstance(content, list)
            assert any(
                "image" in item and "https://example.com/image.jpg" in str(item["image"])
                for item in content
            )
