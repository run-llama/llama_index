import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.zerog import ZeroGLLM


class TestZeroGLLM:
    """Test cases for ZeroGLLM integration."""

    def test_initialization_with_official_model(self):
        """Test initialization with official model."""
        llm = ZeroGLLM(
            model="llama-3.3-70b-instruct",
            private_key="test_private_key"
        )
        
        assert llm.model == "llama-3.3-70b-instruct"
        assert llm.private_key == "test_private_key"
        assert llm.rpc_url == "https://evmrpc-testnet.0g.ai"
        assert llm.context_window == 4096
        assert llm.temperature == 0.1

    def test_initialization_with_custom_provider(self):
        """Test initialization with custom provider."""
        custom_address = "0x1234567890abcdef"
        llm = ZeroGLLM(
            model="custom-model",
            provider_address=custom_address,
            private_key="test_private_key"
        )
        
        assert llm.model == "custom-model"
        assert llm.provider_address == custom_address

    def test_get_provider_address_official_model(self):
        """Test getting provider address for official model."""
        llm = ZeroGLLM(
            model="llama-3.3-70b-instruct",
            private_key="test_private_key"
        )
        
        address = llm._get_provider_address()
        assert address == "0xf07240Efa67755B5311bc75784a061eDB47165Dd"

    def test_get_provider_address_custom_provider(self):
        """Test getting provider address for custom provider."""
        custom_address = "0x1234567890abcdef"
        llm = ZeroGLLM(
            model="custom-model",
            provider_address=custom_address,
            private_key="test_private_key"
        )
        
        address = llm._get_provider_address()
        assert address == custom_address

    def test_get_provider_address_invalid_model(self):
        """Test error handling for invalid model without custom provider."""
        llm = ZeroGLLM(
            model="invalid-model",
            private_key="test_private_key"
        )
        
        with pytest.raises(ValueError, match="Model 'invalid-model' not found"):
            llm._get_provider_address()

    def test_metadata(self):
        """Test LLM metadata."""
        llm = ZeroGLLM(
            model="llama-3.3-70b-instruct",
            private_key="test_private_key",
            context_window=8192,
            max_tokens=1024
        )
        
        metadata = llm.metadata
        assert metadata.context_window == 8192
        assert metadata.num_output == 1024
        assert metadata.is_chat_model is True
        assert metadata.is_function_calling_model is False
        assert metadata.model_name == "llama-3.3-70b-instruct"

    def test_messages_to_openai_format(self):
        """Test message format conversion."""
        llm = ZeroGLLM(
            model="llama-3.3-70b-instruct",
            private_key="test_private_key"
        )
        
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
            ChatMessage(role=MessageRole.USER, content="How are you?")
        ]
        
        openai_messages = llm._messages_to_openai_format(messages)
        
        expected = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        assert openai_messages == expected

    @patch('llama_index.llms.zerog.base.ZeroGLLM._simulate_response')
    @patch('llama_index.llms.zerog.base.ZeroGLLM._initialize_broker')
    def test_chat_sync(self, mock_init_broker, mock_simulate_response):
        """Test synchronous chat functionality."""
        # Setup mocks
        mock_init_broker.return_value = None
        mock_simulate_response.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm doing well, thank you."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }
        
        llm = ZeroGLLM(
            model="llama-3.3-70b-instruct",
            private_key="test_private_key"
        )
        
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello, how are you?")
        ]
        
        with patch('asyncio.new_event_loop') as mock_loop_constructor:
            mock_loop = MagicMock()
            mock_loop_constructor.return_value = mock_loop
            mock_loop.run_until_complete.return_value = mock_simulate_response.return_value
            
            response = llm.chat(messages)
            
            assert response.message.role == MessageRole.ASSISTANT
            assert response.message.content == "Hello! I'm doing well, thank you."

    @pytest.mark.asyncio
    async def test_achat(self):
        """Test asynchronous chat functionality."""
        llm = ZeroGLLM(
            model="llama-3.3-70b-instruct",
            private_key="test_private_key"
        )
        
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello")
        ]
        
        with patch.object(llm, '_initialize_broker', new_callable=AsyncMock) as mock_init:
            with patch.object(llm, '_make_request', new_callable=AsyncMock) as mock_request:
                mock_init.return_value = None
                mock_request.return_value = {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "Hello there!"
                        },
                        "finish_reason": "stop"
                    }]
                }
                
                response = await llm.achat(messages)
                
                assert response.message.role == MessageRole.ASSISTANT
                assert response.message.content == "Hello there!"
                mock_init.assert_called_once()
                mock_request.assert_called_once()

    def test_complete_sync(self):
        """Test synchronous completion functionality."""
        llm = ZeroGLLM(
            model="llama-3.3-70b-instruct",
            private_key="test_private_key"
        )
        
        with patch.object(llm, 'chat') as mock_chat:
            mock_chat.return_value = MagicMock()
            mock_chat.return_value.message.content = "Completion response"
            
            response = llm.complete("Test prompt")
            
            # The complete method should call chat internally
            mock_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_acomplete(self):
        """Test asynchronous completion functionality."""
        llm = ZeroGLLM(
            model="llama-3.3-70b-instruct",
            private_key="test_private_key"
        )
        
        with patch.object(llm, 'achat', new_callable=AsyncMock) as mock_achat:
            mock_response = MagicMock()
            mock_response.message.content = "Async completion response"
            mock_achat.return_value = mock_response
            
            response = await llm.acomplete("Test prompt")
            
            # The acomplete method should call achat internally
            mock_achat.assert_called_once()

    def test_class_name(self):
        """Test class name method."""
        assert ZeroGLLM.class_name() == "ZeroGLLM"

    def test_official_services_constants(self):
        """Test that official services are properly defined."""
        from llama_index.llms.zerog.base import OFFICIAL_0G_SERVICES
        
        assert "llama-3.3-70b-instruct" in OFFICIAL_0G_SERVICES
        assert "deepseek-r1-70b" in OFFICIAL_0G_SERVICES
        
        llama_service = OFFICIAL_0G_SERVICES["llama-3.3-70b-instruct"]
        assert llama_service["provider_address"] == "0xf07240Efa67755B5311bc75784a061eDB47165Dd"
        assert "TEE (TeeML)" in llama_service["verification"]
        
        deepseek_service = OFFICIAL_0G_SERVICES["deepseek-r1-70b"]
        assert deepseek_service["provider_address"] == "0x3feE5a4dd5FDb8a32dDA97Bed899830605dBD9D3"
        assert "TEE (TeeML)" in deepseek_service["verification"]

    @pytest.mark.asyncio
    async def test_simulate_response(self):
        """Test the simulate response method."""
        llm = ZeroGLLM(
            model="llama-3.3-70b-instruct",
            private_key="test_private_key"
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        response = await llm._simulate_response(messages)
        
        assert "choices" in response
        assert len(response["choices"]) == 1
        assert "message" in response["choices"][0]
        assert "content" in response["choices"][0]["message"]
        assert "Hello" in response["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_simulate_streaming_response(self):
        """Test the simulate streaming response method."""
        llm = ZeroGLLM(
            model="llama-3.3-70b-instruct",
            private_key="test_private_key"
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        chunks = []
        
        async for chunk in llm._simulate_streaming_response(messages):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        
        # Check first chunk
        first_chunk = chunks[0]
        assert "choices" in first_chunk
        assert "delta" in first_chunk["choices"][0]
        assert "content" in first_chunk["choices"][0]["delta"]
        
        # Check last chunk has finish_reason
        last_chunk = chunks[-1]
        assert last_chunk["choices"][0]["finish_reason"] == "stop"

    def test_additional_kwargs(self):
        """Test additional kwargs are properly stored."""
        additional_kwargs = {
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2
        }
        
        llm = ZeroGLLM(
            model="llama-3.3-70b-instruct",
            private_key="test_private_key",
            additional_kwargs=additional_kwargs
        )
        
        assert llm.additional_kwargs == additional_kwargs

    def test_custom_parameters(self):
        """Test custom parameters are properly set."""
        llm = ZeroGLLM(
            model="deepseek-r1-70b",
            private_key="test_private_key",
            rpc_url="https://custom-rpc.example.com",
            context_window=8192,
            max_tokens=2048,
            temperature=0.7,
            timeout=120.0
        )
        
        assert llm.model == "deepseek-r1-70b"
        assert llm.rpc_url == "https://custom-rpc.example.com"
        assert llm.context_window == 8192
        assert llm.max_tokens == 2048
        assert llm.temperature == 0.7
        assert llm.timeout == 120.0
