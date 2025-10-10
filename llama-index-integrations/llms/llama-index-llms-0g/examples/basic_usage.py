"""
Basic usage example for 0G Compute Network LLM integration.

This example demonstrates how to use the ZeroGLLM class for basic
chat and completion tasks.
"""

import asyncio
import os
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.zerog import ZeroGLLM


def basic_completion_example():
    """Demonstrate basic completion functionality."""
    print("=== Basic Completion Example ===")
    
    # Initialize the LLM with your private key
    # In production, use environment variables for security
    llm = ZeroGLLM(
        model="llama-3.3-70b-instruct",
        private_key=os.getenv("ETHEREUM_PRIVATE_KEY", "your_private_key_here"),
        temperature=0.7,
        max_tokens=512
    )
    
    # Simple completion
    prompt = "Explain the concept of decentralized computing in simple terms."
    response = llm.complete(prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response.text}")
    print()


def chat_example():
    """Demonstrate chat functionality."""
    print("=== Chat Example ===")
    
    llm = ZeroGLLM(
        model="deepseek-r1-70b",  # Using the reasoning model
        private_key=os.getenv("ETHEREUM_PRIVATE_KEY", "your_private_key_here"),
        temperature=0.3  # Lower temperature for more focused responses
    )
    
    # Create a conversation
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are a helpful AI assistant specialized in blockchain and decentralized technologies."
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="What are the advantages of using a decentralized compute network like 0G?"
        )
    ]
    
    response = llm.chat(messages)
    
    print("Conversation:")
    for msg in messages:
        print(f"{msg.role.value}: {msg.content}")
    
    print(f"Assistant: {response.message.content}")
    print()


def streaming_example():
    """Demonstrate streaming functionality."""
    print("=== Streaming Example ===")
    
    llm = ZeroGLLM(
        model="llama-3.3-70b-instruct",
        private_key=os.getenv("ETHEREUM_PRIVATE_KEY", "your_private_key_here")
    )
    
    prompt = "Write a short story about AI and blockchain technology working together."
    
    print(f"Prompt: {prompt}")
    print("Streaming response:")
    
    # Stream the response
    for chunk in llm.stream_complete(prompt):
        print(chunk.delta, end="", flush=True)
    
    print("\n")


def custom_provider_example():
    """Demonstrate using a custom provider."""
    print("=== Custom Provider Example ===")
    
    # Example with custom provider address
    llm = ZeroGLLM(
        model="custom-model-name",
        provider_address="0x1234567890abcdef1234567890abcdef12345678",  # Example address
        private_key=os.getenv("ETHEREUM_PRIVATE_KEY", "your_private_key_here"),
        rpc_url="https://evmrpc-testnet.0g.ai",
        context_window=8192,
        additional_kwargs={
            "top_p": 0.9,
            "frequency_penalty": 0.1
        }
    )
    
    print(f"Using custom provider: {llm._get_provider_address()}")
    print(f"Model: {llm.model}")
    print(f"Context window: {llm.context_window}")
    print()


async def async_example():
    """Demonstrate async functionality."""
    print("=== Async Example ===")
    
    llm = ZeroGLLM(
        model="llama-3.3-70b-instruct",
        private_key=os.getenv("ETHEREUM_PRIVATE_KEY", "your_private_key_here")
    )
    
    # Async completion
    prompt = "What is the future of decentralized AI?"
    response = await llm.acomplete(prompt)
    
    print(f"Async completion result: {response.text}")
    
    # Async chat
    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello, how are you?")
    ]
    
    chat_response = await llm.achat(messages)
    print(f"Async chat result: {chat_response.message.content}")
    
    # Async streaming
    print("Async streaming:")
    async for chunk in await llm.astream_complete("Tell me about 0G network"):
        print(chunk.delta, end="", flush=True)
    
    print("\n")


def error_handling_example():
    """Demonstrate error handling."""
    print("=== Error Handling Example ===")
    
    try:
        # This should raise an error for invalid model
        llm = ZeroGLLM(
            model="invalid-model-name",
            private_key="test_key"
        )
        
        # This will trigger the error when trying to get provider address
        llm._get_provider_address()
        
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    try:
        # Valid configuration
        llm = ZeroGLLM(
            model="llama-3.3-70b-instruct",
            private_key=os.getenv("ETHEREUM_PRIVATE_KEY", "your_private_key_here")
        )
        
        print(f"Successfully initialized with model: {llm.model}")
        print(f"Provider address: {llm._get_provider_address()}")
        
    except Exception as e:
        print(f"Configuration error: {e}")
    
    print()


def main():
    """Run all examples."""
    print("0G Compute Network LLM Integration Examples")
    print("=" * 50)
    print()
    
    # Check if private key is set
    if not os.getenv("ETHEREUM_PRIVATE_KEY"):
        print("Warning: ETHEREUM_PRIVATE_KEY environment variable not set.")
        print("Using placeholder value for demonstration.")
        print("In production, set your actual private key as an environment variable.")
        print()
    
    # Run synchronous examples
    basic_completion_example()
    chat_example()
    streaming_example()
    custom_provider_example()
    error_handling_example()
    
    # Run async example
    print("Running async example...")
    asyncio.run(async_example())
    
    print("All examples completed!")


if __name__ == "__main__":
    main()

