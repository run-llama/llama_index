"""Basic tests for Fireworks MultiModal."""

from llama_index.multi_modal_llms.fireworks import FireworksMultiModal
from llama_index.core.base.llms.types import ImageBlock, MessageRole
from llama_index.core.schema import ImageNode


def test_init():
    """Test initialization."""
    llm = FireworksMultiModal(api_key="test_key")
    assert llm is not None
    assert llm.api_key == "test_key"


def test_class_name():
    """Test class name."""
    assert FireworksMultiModal.class_name() == "fireworks_multi_modal_llm"


def test_default_model():
    """Test default model."""
    llm = FireworksMultiModal(api_key="test")
    assert "phi-3-vision" in llm.model


def test_custom_model():
    """Test custom model."""
    llm = FireworksMultiModal(model="custom-model", api_key="test")
    assert llm.model == "custom-model"


def test_text_only_message():
    """Test text-only message creation."""
    llm = FireworksMultiModal(api_key="test")
    msg = llm._get_multi_modal_chat_message("Hello", MessageRole.USER.value, [])
    assert msg.content == "Hello"
    assert len(msg.blocks) == 0


def test_message_with_image_block():
    """Test message with ImageBlock."""
    llm = FireworksMultiModal(api_key="test")
    img = ImageBlock(url="http://test.com/img.jpg")
    msg = llm._get_multi_modal_chat_message("Test", MessageRole.USER.value, [img])
    assert len(msg.blocks) == 2


def test_message_with_image_node():
    """Test message with ImageNode."""
    llm = FireworksMultiModal(api_key="test")
    img = ImageNode(image_url="http://test.com/img.jpg")
    msg = llm._get_multi_modal_chat_message("Test", MessageRole.USER.value, [img])
    assert len(msg.blocks) == 2


def test_multiple_images():
    """Test multiple images."""
    llm = FireworksMultiModal(api_key="test")
    imgs = [
        ImageBlock(url="http://test.com/1.jpg"),
        ImageBlock(url="http://test.com/2.jpg"),
    ]
    msg = llm._get_multi_modal_chat_message("Test", MessageRole.USER.value, imgs)
    assert len(msg.blocks) == 3  # 1 text + 2 images
