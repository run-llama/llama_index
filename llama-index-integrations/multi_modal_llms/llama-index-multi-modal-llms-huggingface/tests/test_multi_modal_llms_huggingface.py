import pytest
from unittest.mock import patch, MagicMock
from llama_index.multi_modal_llms.huggingface import HuggingFaceMultiModal
from llama_index.core.schema import ImageDocument

# Mock the model and processor initialization for all test cases
@pytest.fixture(autouse=True)
def mock_huggingface_model_init():
    with patch("transformers.AutoConfig.from_pretrained") as mock_config, \
         patch("transformers.AutoProcessor.from_pretrained") as mock_processor, \
         patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model, \
         patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained") as mock_qwen, \
         patch("transformers.PaliGemmaForConditionalGeneration.from_pretrained") as mock_pali:

        # Mocking return values for each model and processor
        mock_config.return_value = MagicMock(architectures=["Qwen2VLForConditionalGeneration"])
        mock_processor.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_qwen.return_value = MagicMock()
        mock_pali.return_value = MagicMock()

        yield

# Test for PaliGemma model
def test_pali_gemma_model():
    llm = HuggingFaceMultiModal.from_model_name("google/paligemma-3b-pt-224")
    image = ImageDocument(image_path="test_images/man_read.jpg")
    
    with patch.object(llm, '_prepare_messages', return_value={'mock': 'inputs'}), \
         patch.object(llm, '_generate', return_value="This is a test description"):

        result = llm.complete("Describe the image", image_documents=[image])
        assert result.text is not None and result.text != ""
        assert True  # Return True if we receive any response

# Test for Phi model
def test_phi_model():
    llm = HuggingFaceMultiModal.from_model_name("microsoft/Phi-3.5-vision-instruct")
    image = ImageDocument(image_path="test_images/5cats.jpg")
    
    with patch.object(llm, '_prepare_messages', return_value={'mock': 'inputs'}), \
         patch.object(llm, '_generate', return_value="This is a Phi model description"):
        
        result = llm.complete("Describe the image", image_documents=[image])
        assert result.text is not None and result.text != ""
        assert True  # Return True if we receive any response

# Test for Florence with multiple images
def test_florence_model_multiple_images():
    llm = HuggingFaceMultiModal.from_model_name("microsoft/Florence-2-base")
    image1 = ImageDocument(image_path="test_images/5cats.jpg")
    image2 = ImageDocument(image_path="test_images/girl_rabbit.jpg")

    with patch.object(llm, '_prepare_messages', return_value={'mock': 'inputs'}), \
         patch.object(llm, '_generate', return_value="Florence caption with details"):

        result = llm.complete("<DETAILED_CAPTION>", image_documents=[image1, image2])
        assert result.text is not None and result.text != ""
        assert True  # Return True if we receive any response

# Test for Qwen with multiple images
def test_qwen_model_multiple_images():
    llm = HuggingFaceMultiModal.from_model_name("Qwen/Qwen2-VL-2B-Instruct")
    image1 = ImageDocument(image_path="test_images/2dogs.jpg")
    image2 = ImageDocument(image_path="test_images/man_read.jpg")

    with patch.object(llm, '_prepare_messages', return_value={'mock': 'inputs'}), \
         patch.object(llm, '_generate', return_value="Qwen model response for images"):

        result = llm.complete("Describe the images", image_documents=[image1, image2])
        assert result.text is not None and result.text != ""
        assert True  # Return True if we receive any response
